#include "kernels_RVV_16x2_fp32.h"



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
void gemm_RVV_10x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B.data[(k) * (B.strides[0])], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B.data[(k) * (B.strides[0])], A_regt,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(2));
}

// gemm_RVV_10x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 10] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 10] @DRAM
// )
void gemm_RVV_10x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(8));
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[8],(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B.data[(k) * (B.strides[0])], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B.data[(k) * (B.strides[0])], A_regt,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(2));
}

// gemm_RVV_10x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 10] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 10] @DRAM
// )
void gemm_RVV_10x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B.data[(k) * (B.strides[0])], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B.data[(k) * (B.strides[0]) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B.data[(k) * (B.strides[0])], A_regt,(2));
  C_regt_1 = __riscv_vfmacc_vf_f32m1(C_regt_1, B.data[(k) * (B.strides[0]) + 1], A_regt,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(2));
}

// gemm_RVV_10x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 10] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 10] @DRAM
// )
void gemm_RVV_10x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(8));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(8));
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[8],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(2));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B.data[(k) * (B.strides[0])], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B.data[(k) * (B.strides[0]) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B.data[(k) * (B.strides[0])], A_regt,(2));
  C_regt_1 = __riscv_vfmacc_vf_f32m1(C_regt_1, B.data[(k) * (B.strides[0]) + 1], A_regt,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(2));
}

// gemm_RVV_11x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 11] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 11] @DRAM
// )
void gemm_RVV_11x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B.data[(k) * (B.strides[0])], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B.data[(k) * (B.strides[0])], A_regt,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(3));
}

// gemm_RVV_11x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 11] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 11] @DRAM
// )
void gemm_RVV_11x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(8));
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[8],(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B.data[(k) * (B.strides[0])], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B.data[(k) * (B.strides[0])], A_regt,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(3));
}

// gemm_RVV_11x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 11] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 11] @DRAM
// )
void gemm_RVV_11x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B.data[(k) * (B.strides[0])], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B.data[(k) * (B.strides[0]) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B.data[(k) * (B.strides[0])], A_regt,(3));
  C_regt_1 = __riscv_vfmacc_vf_f32m1(C_regt_1, B.data[(k) * (B.strides[0]) + 1], A_regt,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(3));
}

// gemm_RVV_11x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 11] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 11] @DRAM
// )
void gemm_RVV_11x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(8));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(8));
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[8],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(3));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B.data[(k) * (B.strides[0])], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B.data[(k) * (B.strides[0]) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B.data[(k) * (B.strides[0])], A_regt,(3));
  C_regt_1 = __riscv_vfmacc_vf_f32m1(C_regt_1, B.data[(k) * (B.strides[0]) + 1], A_regt,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(3));
}

// gemm_RVV_12x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 12] @DRAM
// )
void gemm_RVV_12x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B.data[(k) * (B.strides[0])], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B.data[(k) * (B.strides[0])], A_regt,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(4));
}

// gemm_RVV_12x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 12] @DRAM
// )
void gemm_RVV_12x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(8));
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B.data[(k) * (B.strides[0])], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B.data[(k) * (B.strides[0])], A_regt,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(4));
}

// gemm_RVV_12x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 12] @DRAM
// )
void gemm_RVV_12x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B.data[(k) * (B.strides[0])], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B.data[(k) * (B.strides[0]) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B.data[(k) * (B.strides[0])], A_regt,(4));
  C_regt_1 = __riscv_vfmacc_vf_f32m1(C_regt_1, B.data[(k) * (B.strides[0]) + 1], A_regt,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(4));
}

// gemm_RVV_12x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 12] @DRAM
// )
void gemm_RVV_12x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(8));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(8));
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[8],(4));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(4));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B.data[(k) * (B.strides[0])], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B.data[(k) * (B.strides[0]) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B.data[(k) * (B.strides[0])], A_regt,(4));
  C_regt_1 = __riscv_vfmacc_vf_f32m1(C_regt_1, B.data[(k) * (B.strides[0]) + 1], A_regt,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(4));
}

// gemm_RVV_13x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 13] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 13] @DRAM
// )
void gemm_RVV_13x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B.data[(k) * (B.strides[0])], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B.data[(k) * (B.strides[0])], A_regt,(5));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(5));
}

// gemm_RVV_13x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 13] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 13] @DRAM
// )
void gemm_RVV_13x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(8));
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[8],(5));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B.data[(k) * (B.strides[0])], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B.data[(k) * (B.strides[0])], A_regt,(5));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(5));
}

// gemm_RVV_13x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 13] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 13] @DRAM
// )
void gemm_RVV_13x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B.data[(k) * (B.strides[0])], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B.data[(k) * (B.strides[0]) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B.data[(k) * (B.strides[0])], A_regt,(5));
  C_regt_1 = __riscv_vfmacc_vf_f32m1(C_regt_1, B.data[(k) * (B.strides[0]) + 1], A_regt,(5));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(5));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(5));
}

// gemm_RVV_13x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 13] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 13] @DRAM
// )
void gemm_RVV_13x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(8));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(8));
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[8],(5));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(5));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(5));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B.data[(k) * (B.strides[0])], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B.data[(k) * (B.strides[0]) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B.data[(k) * (B.strides[0])], A_regt,(5));
  C_regt_1 = __riscv_vfmacc_vf_f32m1(C_regt_1, B.data[(k) * (B.strides[0]) + 1], A_regt,(5));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(5));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(5));
}

// gemm_RVV_14x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 14] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 14] @DRAM
// )
void gemm_RVV_14x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B.data[(k) * (B.strides[0])], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B.data[(k) * (B.strides[0])], A_regt,(6));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(6));
}

// gemm_RVV_14x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 14] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 14] @DRAM
// )
void gemm_RVV_14x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(8));
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[8],(6));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B.data[(k) * (B.strides[0])], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B.data[(k) * (B.strides[0])], A_regt,(6));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(6));
}

// gemm_RVV_14x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 14] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 14] @DRAM
// )
void gemm_RVV_14x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B.data[(k) * (B.strides[0])], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B.data[(k) * (B.strides[0]) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B.data[(k) * (B.strides[0])], A_regt,(6));
  C_regt_1 = __riscv_vfmacc_vf_f32m1(C_regt_1, B.data[(k) * (B.strides[0]) + 1], A_regt,(6));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(6));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(6));
}

// gemm_RVV_14x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 14] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 14] @DRAM
// )
void gemm_RVV_14x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(8));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(8));
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[8],(6));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(6));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(6));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B.data[(k) * (B.strides[0])], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B.data[(k) * (B.strides[0]) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B.data[(k) * (B.strides[0])], A_regt,(6));
  C_regt_1 = __riscv_vfmacc_vf_f32m1(C_regt_1, B.data[(k) * (B.strides[0]) + 1], A_regt,(6));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(6));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(6));
}

// gemm_RVV_15x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 15] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 15] @DRAM
// )
void gemm_RVV_15x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B.data[(k) * (B.strides[0])], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B.data[(k) * (B.strides[0])], A_regt,(7));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(7));
}

// gemm_RVV_15x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 15] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 15] @DRAM
// )
void gemm_RVV_15x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(8));
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[8],(7));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B.data[(k) * (B.strides[0])], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B.data[(k) * (B.strides[0])], A_regt,(7));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(7));
}

// gemm_RVV_15x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 15] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 15] @DRAM
// )
void gemm_RVV_15x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B.data[(k) * (B.strides[0])], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B.data[(k) * (B.strides[0]) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B.data[(k) * (B.strides[0])], A_regt,(7));
  C_regt_1 = __riscv_vfmacc_vf_f32m1(C_regt_1, B.data[(k) * (B.strides[0]) + 1], A_regt,(7));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(7));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(7));
}

// gemm_RVV_15x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 15] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 15] @DRAM
// )
void gemm_RVV_15x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(8));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(8));
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[8],(7));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(7));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(7));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B.data[(k) * (B.strides[0])], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B.data[(k) * (B.strides[0]) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B.data[(k) * (B.strides[0])], A_regt,(7));
  C_regt_1 = __riscv_vfmacc_vf_f32m1(C_regt_1, B.data[(k) * (B.strides[0]) + 1], A_regt,(7));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(7));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(7));
}

// gemm_RVV_16x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 16] @DRAM
// )
void gemm_RVV_16x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B.data[(k) * (B.strides[0])], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B.data[(k) * (B.strides[0])], A_reg_1,(8));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_1,(8));
}

// gemm_RVV_16x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 16] @DRAM
// )
void gemm_RVV_16x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[8],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B.data[(k) * (B.strides[0])], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B.data[(k) * (B.strides[0])], A_reg_1,(8));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_1,(8));
}

// gemm_RVV_16x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 16] @DRAM
// )
void gemm_RVV_16x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B.data[(k) * (B.strides[0])], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B.data[(k) * (B.strides[0])], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B.data[(k) * (B.strides[0]) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f32m1(C_reg_1_1, B.data[(k) * (B.strides[0]) + 1], A_reg_1,(8));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
}

// gemm_RVV_16x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 16] @DRAM
// )
void gemm_RVV_16x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(8));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C.data[8],(8));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(8));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(8));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(8));
  A_reg_1 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(8));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B.data[(k) * (B.strides[0])], A_reg_0,(8));
  C_reg_0_1 = __riscv_vfmacc_vf_f32m1(C_reg_0_1, B.data[(k) * (B.strides[0])], A_reg_1,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B.data[(k) * (B.strides[0]) + 1], A_reg_0,(8));
  C_reg_1_1 = __riscv_vfmacc_vf_f32m1(C_reg_1_1, B.data[(k) * (B.strides[0]) + 1], A_reg_1,(8));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse32_v_f32m1(&C.data[8], C_reg_0_1,(8));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_reg_1_1,(8));
}

// gemm_RVV_1x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 1] @DRAM
// )
void gemm_RVV_1x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
}

// gemm_RVV_1x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 1] @DRAM
// )
void gemm_RVV_1x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(1));
vfloat32m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
}

// gemm_RVV_1x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 1] @DRAM
// )
void gemm_RVV_1x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f32m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
}

// gemm_RVV_1x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 1] @DRAM
// )
void gemm_RVV_1x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(1));
vfloat32m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f32m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
}

// gemm_RVV_2x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 2] @DRAM
// )
void gemm_RVV_2x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
}

// gemm_RVV_2x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 2] @DRAM
// )
void gemm_RVV_2x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(2));
vfloat32m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
}

// gemm_RVV_2x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 2] @DRAM
// )
void gemm_RVV_2x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f32m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
}

// gemm_RVV_2x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 2] @DRAM
// )
void gemm_RVV_2x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(2));
vfloat32m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f32m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
}

// gemm_RVV_3x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 3] @DRAM
// )
void gemm_RVV_3x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
}

// gemm_RVV_3x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 3] @DRAM
// )
void gemm_RVV_3x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(3));
vfloat32m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
}

// gemm_RVV_3x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 3] @DRAM
// )
void gemm_RVV_3x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f32m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
}

// gemm_RVV_3x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 3] @DRAM
// )
void gemm_RVV_3x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(3));
vfloat32m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f32m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
}

// gemm_RVV_4x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 4] @DRAM
// )
void gemm_RVV_4x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
}

// gemm_RVV_4x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 4] @DRAM
// )
void gemm_RVV_4x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
vfloat32m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
}

// gemm_RVV_4x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 4] @DRAM
// )
void gemm_RVV_4x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f32m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
}

// gemm_RVV_4x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 4] @DRAM
// )
void gemm_RVV_4x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
vfloat32m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f32m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
}

// gemm_RVV_5x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 5] @DRAM
// )
void gemm_RVV_5x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
vfloat32m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(5));
}

// gemm_RVV_5x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 5] @DRAM
// )
void gemm_RVV_5x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(5));
vfloat32m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(5));
}

// gemm_RVV_5x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 5] @DRAM
// )
void gemm_RVV_5x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(5));
vfloat32m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f32m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(5));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(5));
}

// gemm_RVV_5x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 5] @DRAM
// )
void gemm_RVV_5x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(5));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(5));
vfloat32m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f32m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(5));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(5));
}

// gemm_RVV_6x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 6] @DRAM
// )
void gemm_RVV_6x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
vfloat32m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(6));
}

// gemm_RVV_6x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 6] @DRAM
// )
void gemm_RVV_6x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(6));
vfloat32m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(6));
}

// gemm_RVV_6x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 6] @DRAM
// )
void gemm_RVV_6x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(6));
vfloat32m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f32m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(6));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(6));
}

// gemm_RVV_6x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 6] @DRAM
// )
void gemm_RVV_6x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(6));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(6));
vfloat32m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f32m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(6));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(6));
}

// gemm_RVV_7x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 7] @DRAM
// )
void gemm_RVV_7x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
vfloat32m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(7));
}

// gemm_RVV_7x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 7] @DRAM
// )
void gemm_RVV_7x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(7));
vfloat32m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(7));
}

// gemm_RVV_7x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 7] @DRAM
// )
void gemm_RVV_7x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(7));
vfloat32m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f32m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(7));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(7));
}

// gemm_RVV_7x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 7] @DRAM
// )
void gemm_RVV_7x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(7));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(7));
vfloat32m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f32m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(7));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(7));
}

// gemm_RVV_8x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 8] @DRAM
// )
void gemm_RVV_8x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
vfloat32m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(8));
}

// gemm_RVV_8x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 8] @DRAM
// )
void gemm_RVV_8x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(8));
vfloat32m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(8));
}

// gemm_RVV_8x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 8] @DRAM
// )
void gemm_RVV_8x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
vfloat32m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f32m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(8));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(8));
}

// gemm_RVV_8x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 8] @DRAM
// )
void gemm_RVV_8x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(8));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(8));
vfloat32m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f32m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f32m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(8));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(8));
}

// gemm_RVV_9x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 9] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 9] @DRAM
// )
void gemm_RVV_9x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B.data[(k) * (B.strides[0])], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B.data[(k) * (B.strides[0])], A_regt,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(1));
}

// gemm_RVV_9x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 9] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 9] @DRAM
// )
void gemm_RVV_9x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(8));
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[8],(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B.data[(k) * (B.strides[0])], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B.data[(k) * (B.strides[0])], A_regt,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(1));
}

// gemm_RVV_9x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 9] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 9] @DRAM
// )
void gemm_RVV_9x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(8));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B.data[(k) * (B.strides[0])], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B.data[(k) * (B.strides[0]) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B.data[(k) * (B.strides[0])], A_regt,(1));
  C_regt_1 = __riscv_vfmacc_vf_f32m1(C_regt_1, B.data[(k) * (B.strides[0]) + 1], A_regt,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(1));
}

// gemm_RVV_9x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 9] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 9] @DRAM
// )
void gemm_RVV_9x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C.data[0],(8));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(8));
C_regt_0 = __riscv_vle32_v_f32m1(&C.data[8],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0] + 8],(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(8));
  A_regt = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0]) + 8],(1));
  C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B.data[(k) * (B.strides[0])], A_reg_0,(8));
  C_reg_1_0 = __riscv_vfmacc_vf_f32m1(C_reg_1_0, B.data[(k) * (B.strides[0]) + 1], A_reg_0,(8));
  C_regt_0 = __riscv_vfmacc_vf_f32m1(C_regt_0, B.data[(k) * (B.strides[0])], A_regt,(1));
  C_regt_1 = __riscv_vfmacc_vf_f32m1(C_regt_1, B.data[(k) * (B.strides[0]) + 1], A_regt,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0_0,(8));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1_0,(8));
__riscv_vse32_v_f32m1(&C.data[8], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0] + 8], C_regt_1,(1));
}


/* relying on the following instruction..."
rvv_broadcast_8xf32_0(dst,vl)
{dst_data} = __riscv_vfmv_v_f_f32m1(0.0f,{vl});
*/

/* relying on the following instruction..."
rvv_vfmacc_8xf32_1xf32(dst,lhs,rhs,vl)
{dst_data} = __riscv_vfmacc_vf_f32m1({dst_data}, {rhs_data}, {lhs_data},{vl});
*/

/* relying on the following instruction..."
rvv_vld_8xf32(dst,src,vl)
{dst_data} = __riscv_vle32_v_f32m1(&{src_data},{vl});
*/

/* relying on the following instruction..."
rvv_vst_8xf32(dst,src,vl)
__riscv_vse32_v_f32m1(&{dst_data}, {src_data},{vl});
*/