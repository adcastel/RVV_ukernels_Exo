#include "kernels_RVV_8x4_fp16.h"



#include <stdio.h>
#include <stdlib.h>

#include <riscv_vector.h>


// gemm_RVV_1x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 1] @DRAM
// )
void gemm_RVV_1x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
}

// gemm_RVV_1x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 1] @DRAM
// )
void gemm_RVV_1x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
}

// gemm_RVV_1x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 1] @DRAM
// )
void gemm_RVV_1x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
}

// gemm_RVV_1x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 1] @DRAM
// )
void gemm_RVV_1x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
}

// gemm_RVV_1x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 1] @DRAM
// )
void gemm_RVV_1x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
}

// gemm_RVV_1x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 1] @DRAM
// )
void gemm_RVV_1x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
}

// gemm_RVV_1x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 1] @DRAM
// )
void gemm_RVV_1x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
}

// gemm_RVV_1x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 1] @DRAM
// )
void gemm_RVV_1x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(1));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(1));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(1));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(1));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(1));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(1));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
}

// gemm_RVV_2x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 2] @DRAM
// )
void gemm_RVV_2x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
}

// gemm_RVV_2x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 2] @DRAM
// )
void gemm_RVV_2x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
}

// gemm_RVV_2x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 2] @DRAM
// )
void gemm_RVV_2x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
}

// gemm_RVV_2x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 2] @DRAM
// )
void gemm_RVV_2x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
}

// gemm_RVV_2x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 2] @DRAM
// )
void gemm_RVV_2x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
}

// gemm_RVV_2x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 2] @DRAM
// )
void gemm_RVV_2x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
}

// gemm_RVV_2x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 2] @DRAM
// )
void gemm_RVV_2x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
}

// gemm_RVV_2x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 2] @DRAM
// )
void gemm_RVV_2x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(2));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(2));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(2));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(2));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(2));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(2));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
}

// gemm_RVV_3x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 3] @DRAM
// )
void gemm_RVV_3x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
}

// gemm_RVV_3x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 3] @DRAM
// )
void gemm_RVV_3x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
}

// gemm_RVV_3x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 3] @DRAM
// )
void gemm_RVV_3x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
}

// gemm_RVV_3x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 3] @DRAM
// )
void gemm_RVV_3x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
}

// gemm_RVV_3x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 3] @DRAM
// )
void gemm_RVV_3x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
}

// gemm_RVV_3x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 3] @DRAM
// )
void gemm_RVV_3x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
}

// gemm_RVV_3x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 3] @DRAM
// )
void gemm_RVV_3x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
}

// gemm_RVV_3x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 3] @DRAM
// )
void gemm_RVV_3x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(3));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(3));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(3));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(3));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(3));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(3));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
}

// gemm_RVV_4x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 4] @DRAM
// )
void gemm_RVV_4x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
}

// gemm_RVV_4x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 4] @DRAM
// )
void gemm_RVV_4x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
}

// gemm_RVV_4x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 4] @DRAM
// )
void gemm_RVV_4x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
}

// gemm_RVV_4x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 4] @DRAM
// )
void gemm_RVV_4x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
}

// gemm_RVV_4x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 4] @DRAM
// )
void gemm_RVV_4x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
}

// gemm_RVV_4x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 4] @DRAM
// )
void gemm_RVV_4x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
}

// gemm_RVV_4x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 4] @DRAM
// )
void gemm_RVV_4x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
}

// gemm_RVV_4x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 4] @DRAM
// )
void gemm_RVV_4x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(4));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(4));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(4));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(4));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(4));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
}

// gemm_RVV_5x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 5] @DRAM
// )
void gemm_RVV_5x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
}

// gemm_RVV_5x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 5] @DRAM
// )
void gemm_RVV_5x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
}

// gemm_RVV_5x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 5] @DRAM
// )
void gemm_RVV_5x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
}

// gemm_RVV_5x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 5] @DRAM
// )
void gemm_RVV_5x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
}

// gemm_RVV_5x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 5] @DRAM
// )
void gemm_RVV_5x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
}

// gemm_RVV_5x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 5] @DRAM
// )
void gemm_RVV_5x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
}

// gemm_RVV_5x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 5] @DRAM
// )
void gemm_RVV_5x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(5));
}

// gemm_RVV_5x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 5] @DRAM
// )
void gemm_RVV_5x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(5));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(5));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(5));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(5));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(5));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(5));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(5));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(5));
}

// gemm_RVV_6x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 6] @DRAM
// )
void gemm_RVV_6x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
}

// gemm_RVV_6x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 6] @DRAM
// )
void gemm_RVV_6x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
}

// gemm_RVV_6x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 6] @DRAM
// )
void gemm_RVV_6x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
}

// gemm_RVV_6x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 6] @DRAM
// )
void gemm_RVV_6x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
}

// gemm_RVV_6x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 6] @DRAM
// )
void gemm_RVV_6x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
}

// gemm_RVV_6x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 6] @DRAM
// )
void gemm_RVV_6x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
}

// gemm_RVV_6x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 6] @DRAM
// )
void gemm_RVV_6x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(6));
}

// gemm_RVV_6x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 6] @DRAM
// )
void gemm_RVV_6x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(6));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(6));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(6));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(6));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(6));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(6));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(6));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(6));
}

// gemm_RVV_7x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 7] @DRAM
// )
void gemm_RVV_7x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
}

// gemm_RVV_7x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 7] @DRAM
// )
void gemm_RVV_7x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
}

// gemm_RVV_7x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 7] @DRAM
// )
void gemm_RVV_7x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
}

// gemm_RVV_7x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 7] @DRAM
// )
void gemm_RVV_7x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
}

// gemm_RVV_7x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 7] @DRAM
// )
void gemm_RVV_7x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
}

// gemm_RVV_7x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 7] @DRAM
// )
void gemm_RVV_7x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
}

// gemm_RVV_7x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 7] @DRAM
// )
void gemm_RVV_7x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(7));
}

// gemm_RVV_7x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 7] @DRAM
// )
void gemm_RVV_7x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(7));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(7));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(7));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(7));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(7));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(7));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(7));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(7));
}

// gemm_RVV_8x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 8] @DRAM
// )
void gemm_RVV_8x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
}

// gemm_RVV_8x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 8] @DRAM
// )
void gemm_RVV_8x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
}

// gemm_RVV_8x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 8] @DRAM
// )
void gemm_RVV_8x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
}

// gemm_RVV_8x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 8] @DRAM
// )
void gemm_RVV_8x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
}

// gemm_RVV_8x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 8] @DRAM
// )
void gemm_RVV_8x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
}

// gemm_RVV_8x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 8] @DRAM
// )
void gemm_RVV_8x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
}

// gemm_RVV_8x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 8] @DRAM
// )
void gemm_RVV_8x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
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
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(8));
}

// gemm_RVV_8x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 8] @DRAM
// )
void gemm_RVV_8x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C.data[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C.data[C.strides[0]],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C.data[(2) * (C.strides[0])],(8));
C_reg_3 = __riscv_vle16_v_f16m1(&C.data[(3) * (C.strides[0])],(8));
vfloat16m1_t A_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A.data[(k) * (A.strides[0])],(8));
  C_reg_0 = __riscv_vfmacc_vf_f16m1(C_reg_0, B.data[(k) * (B.strides[0])], A_reg,(8));
  C_reg_1 = __riscv_vfmacc_vf_f16m1(C_reg_1, B.data[(k) * (B.strides[0]) + 1], A_reg,(8));
  C_reg_2 = __riscv_vfmacc_vf_f16m1(C_reg_2, B.data[(k) * (B.strides[0]) + 2], A_reg,(8));
  C_reg_3 = __riscv_vfmacc_vf_f16m1(C_reg_3, B.data[(k) * (B.strides[0]) + 3], A_reg,(8));
}
__riscv_vse16_v_f16m1(&C.data[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C.data[C.strides[0]], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C.data[(2) * (C.strides[0])], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C.data[(3) * (C.strides[0])], C_reg_3,(8));
}


/* relying on the following instruction..."
rvv_broadcast_8xf16_0(dst,vl)
{dst_data} = __riscv_vfmv_v_f_f16m1(0.0f,{vl});
*/

/* relying on the following instruction..."
rvv_vfmacc_8xf16_1xf16(dst,lhs,rhs,vl)
{dst_data} = __riscv_vfmacc_vf_f16m1({dst_data}, {rhs_data}, {lhs_data},{vl});
*/

/* relying on the following instruction..."
rvv_vld_8xf16(dst,src,vl)
{dst_data} = __riscv_vle16_v_f16m1(&{src_data},{vl});
*/

/* relying on the following instruction..."
rvv_vst_8xf16(dst,src,vl)
__riscv_vse16_v_f16m1(&{dst_data}, {src_data},{vl});
*/
