#include "kernels_RVV_4x20_fp32.h"



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
void gemm_RVV_1x10_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(1));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(1));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(1));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
}

// gemm_RVV_1x10_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 1] @DRAM
// )
void gemm_RVV_1x10_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(1));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(1));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(1));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(1));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(1));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(1));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(1));
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(1));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(1));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(1));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
}

// gemm_RVV_1x11_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 1] @DRAM
// )
void gemm_RVV_1x11_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(1));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(1));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(1));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(1));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
}

// gemm_RVV_1x11_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 1] @DRAM
// )
void gemm_RVV_1x11_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(1));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(1));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(1));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(1));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(1));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(1));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(1));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(1));
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(1));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(1));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(1));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(1));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
}

// gemm_RVV_1x12_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 1] @DRAM
// )
void gemm_RVV_1x12_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(1));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(1));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(1));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(1));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(1));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
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
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(1));
}

// gemm_RVV_1x12_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 1] @DRAM
// )
void gemm_RVV_1x12_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(1));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(1));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(1));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(1));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(1));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(1));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(1));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(1));
C_reg_11 = __riscv_vle32_v_f32m1(&C.data[(11) * (C.strides[0])],(1));
vfloat32m1_t A_reg;
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(1));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(1));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(1));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(1));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(1));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
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
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(1));
}

// gemm_RVV_1x13_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 1] @DRAM
// )
void gemm_RVV_1x13_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(1));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(1));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(1));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(1));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(1));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(1));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(1));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(1));
}

// gemm_RVV_1x13_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 1] @DRAM
// )
void gemm_RVV_1x13_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(1));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(1));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(1));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(1));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(1));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(1));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(1));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(1));
C_reg_11 = __riscv_vle32_v_f32m1(&C.data[(11) * (C.strides[0])],(1));
C_reg_12 = __riscv_vle32_v_f32m1(&C.data[(12) * (C.strides[0])],(1));
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(1));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(1));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(1));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(1));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(1));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(1));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(1));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(1));
}

// gemm_RVV_1x14_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 1] @DRAM
// )
void gemm_RVV_1x14_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(1));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(1));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(1));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(1));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(1));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(1));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(1));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(1));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(1));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(1));
}

// gemm_RVV_1x14_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 1] @DRAM
// )
void gemm_RVV_1x14_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(1));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(1));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(1));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(1));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(1));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(1));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(1));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(1));
C_reg_11 = __riscv_vle32_v_f32m1(&C.data[(11) * (C.strides[0])],(1));
C_reg_12 = __riscv_vle32_v_f32m1(&C.data[(12) * (C.strides[0])],(1));
C_reg_13 = __riscv_vle32_v_f32m1(&C.data[(13) * (C.strides[0])],(1));
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(1));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(1));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(1));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(1));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(1));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(1));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(1));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(1));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(1));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(1));
}

// gemm_RVV_1x15_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 1] @DRAM
// )
void gemm_RVV_1x15_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(1));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(1));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(1));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(1));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(1));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(1));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(1));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(1));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(1));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(1));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(1));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(1));
}

// gemm_RVV_1x15_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 1] @DRAM
// )
void gemm_RVV_1x15_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(1));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(1));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(1));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(1));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(1));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(1));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(1));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(1));
C_reg_11 = __riscv_vle32_v_f32m1(&C.data[(11) * (C.strides[0])],(1));
C_reg_12 = __riscv_vle32_v_f32m1(&C.data[(12) * (C.strides[0])],(1));
C_reg_13 = __riscv_vle32_v_f32m1(&C.data[(13) * (C.strides[0])],(1));
C_reg_14 = __riscv_vle32_v_f32m1(&C.data[(14) * (C.strides[0])],(1));
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(1));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(1));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(1));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(1));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(1));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(1));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(1));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(1));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(1));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(1));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(1));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(1));
}

// gemm_RVV_1x16_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 1] @DRAM
// )
void gemm_RVV_1x16_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(1));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(1));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(1));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(1));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(1));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(1));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(1));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(1));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(1));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
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
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(1));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(1));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(1));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(1));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(1));
}

// gemm_RVV_1x16_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 1] @DRAM
// )
void gemm_RVV_1x16_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(1));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(1));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(1));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(1));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(1));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(1));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(1));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(1));
C_reg_11 = __riscv_vle32_v_f32m1(&C.data[(11) * (C.strides[0])],(1));
C_reg_12 = __riscv_vle32_v_f32m1(&C.data[(12) * (C.strides[0])],(1));
C_reg_13 = __riscv_vle32_v_f32m1(&C.data[(13) * (C.strides[0])],(1));
C_reg_14 = __riscv_vle32_v_f32m1(&C.data[(14) * (C.strides[0])],(1));
C_reg_15 = __riscv_vle32_v_f32m1(&C.data[(15) * (C.strides[0])],(1));
vfloat32m1_t A_reg;
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(1));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(1));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(1));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(1));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(1));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(1));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(1));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(1));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(1));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
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
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(1));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(1));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(1));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(1));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(1));
}

// gemm_RVV_1x17_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][17, 1] @DRAM
// )
void gemm_RVV_1x17_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t C_reg_16;
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
C_reg_16 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
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
vfloat32m1_t B_reg_16;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(1));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(1));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(1));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(1));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(1));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(1));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(1));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(1));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(1));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(1));
  B_reg_16 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 16],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
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
  C_reg_16 = __riscv_vfmacc_vv_f32m1(C_reg_16, A_reg, B_reg_16,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(1));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(1));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(1));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(1));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(1));
__riscv_vse32_v_f32m1(&C.data[(16) * (C.strides[0])], C_reg_16,(1));
}

// gemm_RVV_1x17_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][17, 1] @DRAM
// )
void gemm_RVV_1x17_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t C_reg_16;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(1));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(1));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(1));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(1));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(1));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(1));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(1));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(1));
C_reg_11 = __riscv_vle32_v_f32m1(&C.data[(11) * (C.strides[0])],(1));
C_reg_12 = __riscv_vle32_v_f32m1(&C.data[(12) * (C.strides[0])],(1));
C_reg_13 = __riscv_vle32_v_f32m1(&C.data[(13) * (C.strides[0])],(1));
C_reg_14 = __riscv_vle32_v_f32m1(&C.data[(14) * (C.strides[0])],(1));
C_reg_15 = __riscv_vle32_v_f32m1(&C.data[(15) * (C.strides[0])],(1));
C_reg_16 = __riscv_vle32_v_f32m1(&C.data[(16) * (C.strides[0])],(1));
vfloat32m1_t A_reg;
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
vfloat32m1_t B_reg_16;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(1));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(1));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(1));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(1));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(1));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(1));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(1));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(1));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(1));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(1));
  B_reg_16 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 16],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
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
  C_reg_16 = __riscv_vfmacc_vv_f32m1(C_reg_16, A_reg, B_reg_16,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(1));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(1));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(1));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(1));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(1));
__riscv_vse32_v_f32m1(&C.data[(16) * (C.strides[0])], C_reg_16,(1));
}

// gemm_RVV_1x18_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][18, 1] @DRAM
// )
void gemm_RVV_1x18_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t C_reg_16;
vfloat32m1_t C_reg_17;
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
C_reg_16 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_17 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
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
vfloat32m1_t B_reg_16;
vfloat32m1_t B_reg_17;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(1));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(1));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(1));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(1));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(1));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(1));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(1));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(1));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(1));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(1));
  B_reg_16 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 16],(1));
  B_reg_17 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 17],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
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
  C_reg_16 = __riscv_vfmacc_vv_f32m1(C_reg_16, A_reg, B_reg_16,(1));
  C_reg_17 = __riscv_vfmacc_vv_f32m1(C_reg_17, A_reg, B_reg_17,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(1));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(1));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(1));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(1));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(1));
__riscv_vse32_v_f32m1(&C.data[(16) * (C.strides[0])], C_reg_16,(1));
__riscv_vse32_v_f32m1(&C.data[(17) * (C.strides[0])], C_reg_17,(1));
}

// gemm_RVV_1x18_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][18, 1] @DRAM
// )
void gemm_RVV_1x18_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t C_reg_16;
vfloat32m1_t C_reg_17;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(1));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(1));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(1));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(1));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(1));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(1));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(1));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(1));
C_reg_11 = __riscv_vle32_v_f32m1(&C.data[(11) * (C.strides[0])],(1));
C_reg_12 = __riscv_vle32_v_f32m1(&C.data[(12) * (C.strides[0])],(1));
C_reg_13 = __riscv_vle32_v_f32m1(&C.data[(13) * (C.strides[0])],(1));
C_reg_14 = __riscv_vle32_v_f32m1(&C.data[(14) * (C.strides[0])],(1));
C_reg_15 = __riscv_vle32_v_f32m1(&C.data[(15) * (C.strides[0])],(1));
C_reg_16 = __riscv_vle32_v_f32m1(&C.data[(16) * (C.strides[0])],(1));
C_reg_17 = __riscv_vle32_v_f32m1(&C.data[(17) * (C.strides[0])],(1));
vfloat32m1_t A_reg;
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
vfloat32m1_t B_reg_16;
vfloat32m1_t B_reg_17;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(1));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(1));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(1));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(1));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(1));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(1));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(1));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(1));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(1));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(1));
  B_reg_16 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 16],(1));
  B_reg_17 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 17],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
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
  C_reg_16 = __riscv_vfmacc_vv_f32m1(C_reg_16, A_reg, B_reg_16,(1));
  C_reg_17 = __riscv_vfmacc_vv_f32m1(C_reg_17, A_reg, B_reg_17,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(1));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(1));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(1));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(1));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(1));
__riscv_vse32_v_f32m1(&C.data[(16) * (C.strides[0])], C_reg_16,(1));
__riscv_vse32_v_f32m1(&C.data[(17) * (C.strides[0])], C_reg_17,(1));
}

// gemm_RVV_1x19_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][19, 1] @DRAM
// )
void gemm_RVV_1x19_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t C_reg_16;
vfloat32m1_t C_reg_17;
vfloat32m1_t C_reg_18;
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
C_reg_16 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_17 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_18 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
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
vfloat32m1_t B_reg_16;
vfloat32m1_t B_reg_17;
vfloat32m1_t B_reg_18;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(1));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(1));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(1));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(1));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(1));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(1));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(1));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(1));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(1));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(1));
  B_reg_16 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 16],(1));
  B_reg_17 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 17],(1));
  B_reg_18 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 18],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
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
  C_reg_16 = __riscv_vfmacc_vv_f32m1(C_reg_16, A_reg, B_reg_16,(1));
  C_reg_17 = __riscv_vfmacc_vv_f32m1(C_reg_17, A_reg, B_reg_17,(1));
  C_reg_18 = __riscv_vfmacc_vv_f32m1(C_reg_18, A_reg, B_reg_18,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(1));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(1));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(1));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(1));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(1));
__riscv_vse32_v_f32m1(&C.data[(16) * (C.strides[0])], C_reg_16,(1));
__riscv_vse32_v_f32m1(&C.data[(17) * (C.strides[0])], C_reg_17,(1));
__riscv_vse32_v_f32m1(&C.data[(18) * (C.strides[0])], C_reg_18,(1));
}

// gemm_RVV_1x19_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][19, 1] @DRAM
// )
void gemm_RVV_1x19_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t C_reg_16;
vfloat32m1_t C_reg_17;
vfloat32m1_t C_reg_18;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(1));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(1));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(1));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(1));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(1));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(1));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(1));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(1));
C_reg_11 = __riscv_vle32_v_f32m1(&C.data[(11) * (C.strides[0])],(1));
C_reg_12 = __riscv_vle32_v_f32m1(&C.data[(12) * (C.strides[0])],(1));
C_reg_13 = __riscv_vle32_v_f32m1(&C.data[(13) * (C.strides[0])],(1));
C_reg_14 = __riscv_vle32_v_f32m1(&C.data[(14) * (C.strides[0])],(1));
C_reg_15 = __riscv_vle32_v_f32m1(&C.data[(15) * (C.strides[0])],(1));
C_reg_16 = __riscv_vle32_v_f32m1(&C.data[(16) * (C.strides[0])],(1));
C_reg_17 = __riscv_vle32_v_f32m1(&C.data[(17) * (C.strides[0])],(1));
C_reg_18 = __riscv_vle32_v_f32m1(&C.data[(18) * (C.strides[0])],(1));
vfloat32m1_t A_reg;
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
vfloat32m1_t B_reg_16;
vfloat32m1_t B_reg_17;
vfloat32m1_t B_reg_18;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(1));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(1));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(1));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(1));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(1));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(1));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(1));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(1));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(1));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(1));
  B_reg_16 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 16],(1));
  B_reg_17 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 17],(1));
  B_reg_18 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 18],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
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
  C_reg_16 = __riscv_vfmacc_vv_f32m1(C_reg_16, A_reg, B_reg_16,(1));
  C_reg_17 = __riscv_vfmacc_vv_f32m1(C_reg_17, A_reg, B_reg_17,(1));
  C_reg_18 = __riscv_vfmacc_vv_f32m1(C_reg_18, A_reg, B_reg_18,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(1));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(1));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(1));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(1));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(1));
__riscv_vse32_v_f32m1(&C.data[(16) * (C.strides[0])], C_reg_16,(1));
__riscv_vse32_v_f32m1(&C.data[(17) * (C.strides[0])], C_reg_17,(1));
__riscv_vse32_v_f32m1(&C.data[(18) * (C.strides[0])], C_reg_18,(1));
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
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
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
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
}

// gemm_RVV_1x20_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 1] @DRAM
// )
void gemm_RVV_1x20_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t C_reg_16;
vfloat32m1_t C_reg_17;
vfloat32m1_t C_reg_18;
vfloat32m1_t C_reg_19;
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
C_reg_16 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_17 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_18 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_19 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
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
vfloat32m1_t B_reg_16;
vfloat32m1_t B_reg_17;
vfloat32m1_t B_reg_18;
vfloat32m1_t B_reg_19;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(1));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(1));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(1));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(1));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(1));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(1));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(1));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(1));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(1));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(1));
  B_reg_16 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 16],(1));
  B_reg_17 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 17],(1));
  B_reg_18 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 18],(1));
  B_reg_19 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 19],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
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
  C_reg_16 = __riscv_vfmacc_vv_f32m1(C_reg_16, A_reg, B_reg_16,(1));
  C_reg_17 = __riscv_vfmacc_vv_f32m1(C_reg_17, A_reg, B_reg_17,(1));
  C_reg_18 = __riscv_vfmacc_vv_f32m1(C_reg_18, A_reg, B_reg_18,(1));
  C_reg_19 = __riscv_vfmacc_vv_f32m1(C_reg_19, A_reg, B_reg_19,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(1));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(1));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(1));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(1));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(1));
__riscv_vse32_v_f32m1(&C.data[(16) * (C.strides[0])], C_reg_16,(1));
__riscv_vse32_v_f32m1(&C.data[(17) * (C.strides[0])], C_reg_17,(1));
__riscv_vse32_v_f32m1(&C.data[(18) * (C.strides[0])], C_reg_18,(1));
__riscv_vse32_v_f32m1(&C.data[(19) * (C.strides[0])], C_reg_19,(1));
}

// gemm_RVV_1x20_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 1] @DRAM
// )
void gemm_RVV_1x20_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t C_reg_16;
vfloat32m1_t C_reg_17;
vfloat32m1_t C_reg_18;
vfloat32m1_t C_reg_19;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(1));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(1));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(1));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(1));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(1));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(1));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(1));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(1));
C_reg_11 = __riscv_vle32_v_f32m1(&C.data[(11) * (C.strides[0])],(1));
C_reg_12 = __riscv_vle32_v_f32m1(&C.data[(12) * (C.strides[0])],(1));
C_reg_13 = __riscv_vle32_v_f32m1(&C.data[(13) * (C.strides[0])],(1));
C_reg_14 = __riscv_vle32_v_f32m1(&C.data[(14) * (C.strides[0])],(1));
C_reg_15 = __riscv_vle32_v_f32m1(&C.data[(15) * (C.strides[0])],(1));
C_reg_16 = __riscv_vle32_v_f32m1(&C.data[(16) * (C.strides[0])],(1));
C_reg_17 = __riscv_vle32_v_f32m1(&C.data[(17) * (C.strides[0])],(1));
C_reg_18 = __riscv_vle32_v_f32m1(&C.data[(18) * (C.strides[0])],(1));
C_reg_19 = __riscv_vle32_v_f32m1(&C.data[(19) * (C.strides[0])],(1));
vfloat32m1_t A_reg;
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
vfloat32m1_t B_reg_16;
vfloat32m1_t B_reg_17;
vfloat32m1_t B_reg_18;
vfloat32m1_t B_reg_19;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(1));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(1));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(1));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(1));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(1));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(1));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(1));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(1));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(1));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(1));
  B_reg_16 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 16],(1));
  B_reg_17 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 17],(1));
  B_reg_18 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 18],(1));
  B_reg_19 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 19],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
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
  C_reg_16 = __riscv_vfmacc_vv_f32m1(C_reg_16, A_reg, B_reg_16,(1));
  C_reg_17 = __riscv_vfmacc_vv_f32m1(C_reg_17, A_reg, B_reg_17,(1));
  C_reg_18 = __riscv_vfmacc_vv_f32m1(C_reg_18, A_reg, B_reg_18,(1));
  C_reg_19 = __riscv_vfmacc_vv_f32m1(C_reg_19, A_reg, B_reg_19,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(1));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(1));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(1));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(1));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(1));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(1));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(1));
__riscv_vse32_v_f32m1(&C.data[(16) * (C.strides[0])], C_reg_16,(1));
__riscv_vse32_v_f32m1(&C.data[(17) * (C.strides[0])], C_reg_17,(1));
__riscv_vse32_v_f32m1(&C.data[(18) * (C.strides[0])], C_reg_18,(1));
__riscv_vse32_v_f32m1(&C.data[(19) * (C.strides[0])], C_reg_19,(1));
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
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
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
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
}

// gemm_RVV_1x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 1] @DRAM
// )
void gemm_RVV_1x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
}

// gemm_RVV_1x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 1] @DRAM
// )
void gemm_RVV_1x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
}

// gemm_RVV_1x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 1] @DRAM
// )
void gemm_RVV_1x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
}

// gemm_RVV_1x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 1] @DRAM
// )
void gemm_RVV_1x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
}

// gemm_RVV_1x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 1] @DRAM
// )
void gemm_RVV_1x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
}

// gemm_RVV_1x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 1] @DRAM
// )
void gemm_RVV_1x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(1));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
}

// gemm_RVV_1x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 1] @DRAM
// )
void gemm_RVV_1x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
}

// gemm_RVV_1x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 1] @DRAM
// )
void gemm_RVV_1x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(1));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(1));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
}

// gemm_RVV_1x7_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 1] @DRAM
// )
void gemm_RVV_1x7_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
}

// gemm_RVV_1x7_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 1] @DRAM
// )
void gemm_RVV_1x7_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(1));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(1));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(1));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
}

// gemm_RVV_1x8_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 1] @DRAM
// )
void gemm_RVV_1x8_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(1));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
}

// gemm_RVV_1x8_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 1] @DRAM
// )
void gemm_RVV_1x8_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(1));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(1));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(1));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(1));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(1));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
}

// gemm_RVV_1x9_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 1] @DRAM
// )
void gemm_RVV_1x9_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_8 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(1));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(1));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
}

// gemm_RVV_1x9_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 1] @DRAM
// )
void gemm_RVV_1x9_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(1));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(1));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(1));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(1));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(1));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(1));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(1));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(1));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(1));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(1));
}

// gemm_RVV_2x10_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 2] @DRAM
// )
void gemm_RVV_2x10_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(2));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(2));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(2));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
}

// gemm_RVV_2x10_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 2] @DRAM
// )
void gemm_RVV_2x10_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(2));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(2));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(2));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(2));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(2));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(2));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(2));
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(2));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(2));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(2));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
}

// gemm_RVV_2x11_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 2] @DRAM
// )
void gemm_RVV_2x11_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(2));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(2));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(2));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(2));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
}

// gemm_RVV_2x11_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 2] @DRAM
// )
void gemm_RVV_2x11_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(2));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(2));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(2));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(2));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(2));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(2));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(2));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(2));
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(2));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(2));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(2));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(2));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
}

// gemm_RVV_2x12_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 2] @DRAM
// )
void gemm_RVV_2x12_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(2));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(2));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(2));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(2));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(2));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
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
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(2));
}

// gemm_RVV_2x12_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 2] @DRAM
// )
void gemm_RVV_2x12_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(2));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(2));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(2));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(2));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(2));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(2));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(2));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(2));
C_reg_11 = __riscv_vle32_v_f32m1(&C.data[(11) * (C.strides[0])],(2));
vfloat32m1_t A_reg;
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(2));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(2));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(2));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(2));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(2));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
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
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(2));
}

// gemm_RVV_2x13_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 2] @DRAM
// )
void gemm_RVV_2x13_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(2));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(2));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(2));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(2));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(2));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(2));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(2));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(2));
}

// gemm_RVV_2x13_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 2] @DRAM
// )
void gemm_RVV_2x13_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(2));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(2));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(2));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(2));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(2));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(2));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(2));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(2));
C_reg_11 = __riscv_vle32_v_f32m1(&C.data[(11) * (C.strides[0])],(2));
C_reg_12 = __riscv_vle32_v_f32m1(&C.data[(12) * (C.strides[0])],(2));
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(2));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(2));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(2));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(2));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(2));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(2));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(2));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(2));
}

// gemm_RVV_2x14_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 2] @DRAM
// )
void gemm_RVV_2x14_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(2));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(2));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(2));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(2));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(2));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(2));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(2));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(2));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(2));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(2));
}

// gemm_RVV_2x14_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 2] @DRAM
// )
void gemm_RVV_2x14_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(2));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(2));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(2));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(2));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(2));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(2));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(2));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(2));
C_reg_11 = __riscv_vle32_v_f32m1(&C.data[(11) * (C.strides[0])],(2));
C_reg_12 = __riscv_vle32_v_f32m1(&C.data[(12) * (C.strides[0])],(2));
C_reg_13 = __riscv_vle32_v_f32m1(&C.data[(13) * (C.strides[0])],(2));
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(2));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(2));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(2));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(2));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(2));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(2));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(2));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(2));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(2));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(2));
}

// gemm_RVV_2x15_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 2] @DRAM
// )
void gemm_RVV_2x15_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(2));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(2));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(2));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(2));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(2));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(2));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(2));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(2));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(2));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(2));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(2));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(2));
}

// gemm_RVV_2x15_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 2] @DRAM
// )
void gemm_RVV_2x15_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(2));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(2));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(2));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(2));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(2));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(2));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(2));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(2));
C_reg_11 = __riscv_vle32_v_f32m1(&C.data[(11) * (C.strides[0])],(2));
C_reg_12 = __riscv_vle32_v_f32m1(&C.data[(12) * (C.strides[0])],(2));
C_reg_13 = __riscv_vle32_v_f32m1(&C.data[(13) * (C.strides[0])],(2));
C_reg_14 = __riscv_vle32_v_f32m1(&C.data[(14) * (C.strides[0])],(2));
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(2));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(2));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(2));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(2));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(2));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(2));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(2));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(2));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(2));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(2));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(2));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(2));
}

// gemm_RVV_2x16_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 2] @DRAM
// )
void gemm_RVV_2x16_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(2));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(2));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(2));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(2));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(2));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(2));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(2));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(2));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(2));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
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
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(2));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(2));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(2));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(2));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(2));
}

// gemm_RVV_2x16_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 2] @DRAM
// )
void gemm_RVV_2x16_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(2));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(2));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(2));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(2));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(2));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(2));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(2));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(2));
C_reg_11 = __riscv_vle32_v_f32m1(&C.data[(11) * (C.strides[0])],(2));
C_reg_12 = __riscv_vle32_v_f32m1(&C.data[(12) * (C.strides[0])],(2));
C_reg_13 = __riscv_vle32_v_f32m1(&C.data[(13) * (C.strides[0])],(2));
C_reg_14 = __riscv_vle32_v_f32m1(&C.data[(14) * (C.strides[0])],(2));
C_reg_15 = __riscv_vle32_v_f32m1(&C.data[(15) * (C.strides[0])],(2));
vfloat32m1_t A_reg;
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(2));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(2));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(2));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(2));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(2));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(2));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(2));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(2));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(2));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
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
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(2));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(2));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(2));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(2));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(2));
}

// gemm_RVV_2x17_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][17, 2] @DRAM
// )
void gemm_RVV_2x17_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t C_reg_16;
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
C_reg_16 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
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
vfloat32m1_t B_reg_16;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(2));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(2));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(2));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(2));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(2));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(2));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(2));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(2));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(2));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(2));
  B_reg_16 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 16],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
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
  C_reg_16 = __riscv_vfmacc_vv_f32m1(C_reg_16, A_reg, B_reg_16,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(2));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(2));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(2));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(2));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(2));
__riscv_vse32_v_f32m1(&C.data[(16) * (C.strides[0])], C_reg_16,(2));
}

// gemm_RVV_2x17_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][17, 2] @DRAM
// )
void gemm_RVV_2x17_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t C_reg_16;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(2));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(2));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(2));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(2));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(2));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(2));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(2));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(2));
C_reg_11 = __riscv_vle32_v_f32m1(&C.data[(11) * (C.strides[0])],(2));
C_reg_12 = __riscv_vle32_v_f32m1(&C.data[(12) * (C.strides[0])],(2));
C_reg_13 = __riscv_vle32_v_f32m1(&C.data[(13) * (C.strides[0])],(2));
C_reg_14 = __riscv_vle32_v_f32m1(&C.data[(14) * (C.strides[0])],(2));
C_reg_15 = __riscv_vle32_v_f32m1(&C.data[(15) * (C.strides[0])],(2));
C_reg_16 = __riscv_vle32_v_f32m1(&C.data[(16) * (C.strides[0])],(2));
vfloat32m1_t A_reg;
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
vfloat32m1_t B_reg_16;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(2));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(2));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(2));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(2));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(2));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(2));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(2));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(2));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(2));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(2));
  B_reg_16 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 16],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
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
  C_reg_16 = __riscv_vfmacc_vv_f32m1(C_reg_16, A_reg, B_reg_16,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(2));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(2));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(2));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(2));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(2));
__riscv_vse32_v_f32m1(&C.data[(16) * (C.strides[0])], C_reg_16,(2));
}

// gemm_RVV_2x18_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][18, 2] @DRAM
// )
void gemm_RVV_2x18_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t C_reg_16;
vfloat32m1_t C_reg_17;
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
C_reg_16 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_17 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
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
vfloat32m1_t B_reg_16;
vfloat32m1_t B_reg_17;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(2));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(2));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(2));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(2));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(2));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(2));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(2));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(2));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(2));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(2));
  B_reg_16 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 16],(2));
  B_reg_17 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 17],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
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
  C_reg_16 = __riscv_vfmacc_vv_f32m1(C_reg_16, A_reg, B_reg_16,(2));
  C_reg_17 = __riscv_vfmacc_vv_f32m1(C_reg_17, A_reg, B_reg_17,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(2));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(2));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(2));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(2));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(2));
__riscv_vse32_v_f32m1(&C.data[(16) * (C.strides[0])], C_reg_16,(2));
__riscv_vse32_v_f32m1(&C.data[(17) * (C.strides[0])], C_reg_17,(2));
}

// gemm_RVV_2x18_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][18, 2] @DRAM
// )
void gemm_RVV_2x18_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t C_reg_16;
vfloat32m1_t C_reg_17;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(2));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(2));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(2));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(2));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(2));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(2));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(2));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(2));
C_reg_11 = __riscv_vle32_v_f32m1(&C.data[(11) * (C.strides[0])],(2));
C_reg_12 = __riscv_vle32_v_f32m1(&C.data[(12) * (C.strides[0])],(2));
C_reg_13 = __riscv_vle32_v_f32m1(&C.data[(13) * (C.strides[0])],(2));
C_reg_14 = __riscv_vle32_v_f32m1(&C.data[(14) * (C.strides[0])],(2));
C_reg_15 = __riscv_vle32_v_f32m1(&C.data[(15) * (C.strides[0])],(2));
C_reg_16 = __riscv_vle32_v_f32m1(&C.data[(16) * (C.strides[0])],(2));
C_reg_17 = __riscv_vle32_v_f32m1(&C.data[(17) * (C.strides[0])],(2));
vfloat32m1_t A_reg;
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
vfloat32m1_t B_reg_16;
vfloat32m1_t B_reg_17;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(2));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(2));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(2));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(2));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(2));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(2));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(2));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(2));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(2));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(2));
  B_reg_16 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 16],(2));
  B_reg_17 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 17],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
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
  C_reg_16 = __riscv_vfmacc_vv_f32m1(C_reg_16, A_reg, B_reg_16,(2));
  C_reg_17 = __riscv_vfmacc_vv_f32m1(C_reg_17, A_reg, B_reg_17,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(2));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(2));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(2));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(2));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(2));
__riscv_vse32_v_f32m1(&C.data[(16) * (C.strides[0])], C_reg_16,(2));
__riscv_vse32_v_f32m1(&C.data[(17) * (C.strides[0])], C_reg_17,(2));
}

// gemm_RVV_2x19_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][19, 2] @DRAM
// )
void gemm_RVV_2x19_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t C_reg_16;
vfloat32m1_t C_reg_17;
vfloat32m1_t C_reg_18;
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
C_reg_16 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_17 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_18 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
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
vfloat32m1_t B_reg_16;
vfloat32m1_t B_reg_17;
vfloat32m1_t B_reg_18;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(2));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(2));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(2));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(2));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(2));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(2));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(2));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(2));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(2));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(2));
  B_reg_16 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 16],(2));
  B_reg_17 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 17],(2));
  B_reg_18 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 18],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
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
  C_reg_16 = __riscv_vfmacc_vv_f32m1(C_reg_16, A_reg, B_reg_16,(2));
  C_reg_17 = __riscv_vfmacc_vv_f32m1(C_reg_17, A_reg, B_reg_17,(2));
  C_reg_18 = __riscv_vfmacc_vv_f32m1(C_reg_18, A_reg, B_reg_18,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(2));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(2));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(2));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(2));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(2));
__riscv_vse32_v_f32m1(&C.data[(16) * (C.strides[0])], C_reg_16,(2));
__riscv_vse32_v_f32m1(&C.data[(17) * (C.strides[0])], C_reg_17,(2));
__riscv_vse32_v_f32m1(&C.data[(18) * (C.strides[0])], C_reg_18,(2));
}

// gemm_RVV_2x19_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][19, 2] @DRAM
// )
void gemm_RVV_2x19_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t C_reg_16;
vfloat32m1_t C_reg_17;
vfloat32m1_t C_reg_18;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(2));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(2));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(2));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(2));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(2));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(2));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(2));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(2));
C_reg_11 = __riscv_vle32_v_f32m1(&C.data[(11) * (C.strides[0])],(2));
C_reg_12 = __riscv_vle32_v_f32m1(&C.data[(12) * (C.strides[0])],(2));
C_reg_13 = __riscv_vle32_v_f32m1(&C.data[(13) * (C.strides[0])],(2));
C_reg_14 = __riscv_vle32_v_f32m1(&C.data[(14) * (C.strides[0])],(2));
C_reg_15 = __riscv_vle32_v_f32m1(&C.data[(15) * (C.strides[0])],(2));
C_reg_16 = __riscv_vle32_v_f32m1(&C.data[(16) * (C.strides[0])],(2));
C_reg_17 = __riscv_vle32_v_f32m1(&C.data[(17) * (C.strides[0])],(2));
C_reg_18 = __riscv_vle32_v_f32m1(&C.data[(18) * (C.strides[0])],(2));
vfloat32m1_t A_reg;
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
vfloat32m1_t B_reg_16;
vfloat32m1_t B_reg_17;
vfloat32m1_t B_reg_18;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(2));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(2));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(2));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(2));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(2));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(2));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(2));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(2));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(2));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(2));
  B_reg_16 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 16],(2));
  B_reg_17 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 17],(2));
  B_reg_18 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 18],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
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
  C_reg_16 = __riscv_vfmacc_vv_f32m1(C_reg_16, A_reg, B_reg_16,(2));
  C_reg_17 = __riscv_vfmacc_vv_f32m1(C_reg_17, A_reg, B_reg_17,(2));
  C_reg_18 = __riscv_vfmacc_vv_f32m1(C_reg_18, A_reg, B_reg_18,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(2));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(2));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(2));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(2));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(2));
__riscv_vse32_v_f32m1(&C.data[(16) * (C.strides[0])], C_reg_16,(2));
__riscv_vse32_v_f32m1(&C.data[(17) * (C.strides[0])], C_reg_17,(2));
__riscv_vse32_v_f32m1(&C.data[(18) * (C.strides[0])], C_reg_18,(2));
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
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
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
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
}

// gemm_RVV_2x20_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 2] @DRAM
// )
void gemm_RVV_2x20_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t C_reg_16;
vfloat32m1_t C_reg_17;
vfloat32m1_t C_reg_18;
vfloat32m1_t C_reg_19;
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
C_reg_16 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_17 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_18 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_19 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
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
vfloat32m1_t B_reg_16;
vfloat32m1_t B_reg_17;
vfloat32m1_t B_reg_18;
vfloat32m1_t B_reg_19;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(2));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(2));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(2));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(2));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(2));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(2));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(2));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(2));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(2));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(2));
  B_reg_16 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 16],(2));
  B_reg_17 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 17],(2));
  B_reg_18 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 18],(2));
  B_reg_19 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 19],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
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
  C_reg_16 = __riscv_vfmacc_vv_f32m1(C_reg_16, A_reg, B_reg_16,(2));
  C_reg_17 = __riscv_vfmacc_vv_f32m1(C_reg_17, A_reg, B_reg_17,(2));
  C_reg_18 = __riscv_vfmacc_vv_f32m1(C_reg_18, A_reg, B_reg_18,(2));
  C_reg_19 = __riscv_vfmacc_vv_f32m1(C_reg_19, A_reg, B_reg_19,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(2));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(2));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(2));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(2));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(2));
__riscv_vse32_v_f32m1(&C.data[(16) * (C.strides[0])], C_reg_16,(2));
__riscv_vse32_v_f32m1(&C.data[(17) * (C.strides[0])], C_reg_17,(2));
__riscv_vse32_v_f32m1(&C.data[(18) * (C.strides[0])], C_reg_18,(2));
__riscv_vse32_v_f32m1(&C.data[(19) * (C.strides[0])], C_reg_19,(2));
}

// gemm_RVV_2x20_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 2] @DRAM
// )
void gemm_RVV_2x20_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t C_reg_16;
vfloat32m1_t C_reg_17;
vfloat32m1_t C_reg_18;
vfloat32m1_t C_reg_19;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(2));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(2));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(2));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(2));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(2));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(2));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(2));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(2));
C_reg_11 = __riscv_vle32_v_f32m1(&C.data[(11) * (C.strides[0])],(2));
C_reg_12 = __riscv_vle32_v_f32m1(&C.data[(12) * (C.strides[0])],(2));
C_reg_13 = __riscv_vle32_v_f32m1(&C.data[(13) * (C.strides[0])],(2));
C_reg_14 = __riscv_vle32_v_f32m1(&C.data[(14) * (C.strides[0])],(2));
C_reg_15 = __riscv_vle32_v_f32m1(&C.data[(15) * (C.strides[0])],(2));
C_reg_16 = __riscv_vle32_v_f32m1(&C.data[(16) * (C.strides[0])],(2));
C_reg_17 = __riscv_vle32_v_f32m1(&C.data[(17) * (C.strides[0])],(2));
C_reg_18 = __riscv_vle32_v_f32m1(&C.data[(18) * (C.strides[0])],(2));
C_reg_19 = __riscv_vle32_v_f32m1(&C.data[(19) * (C.strides[0])],(2));
vfloat32m1_t A_reg;
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
vfloat32m1_t B_reg_16;
vfloat32m1_t B_reg_17;
vfloat32m1_t B_reg_18;
vfloat32m1_t B_reg_19;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(2));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(2));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(2));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(2));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(2));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(2));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(2));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(2));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(2));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(2));
  B_reg_16 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 16],(2));
  B_reg_17 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 17],(2));
  B_reg_18 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 18],(2));
  B_reg_19 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 19],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
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
  C_reg_16 = __riscv_vfmacc_vv_f32m1(C_reg_16, A_reg, B_reg_16,(2));
  C_reg_17 = __riscv_vfmacc_vv_f32m1(C_reg_17, A_reg, B_reg_17,(2));
  C_reg_18 = __riscv_vfmacc_vv_f32m1(C_reg_18, A_reg, B_reg_18,(2));
  C_reg_19 = __riscv_vfmacc_vv_f32m1(C_reg_19, A_reg, B_reg_19,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(2));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(2));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(2));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(2));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(2));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(2));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(2));
__riscv_vse32_v_f32m1(&C.data[(16) * (C.strides[0])], C_reg_16,(2));
__riscv_vse32_v_f32m1(&C.data[(17) * (C.strides[0])], C_reg_17,(2));
__riscv_vse32_v_f32m1(&C.data[(18) * (C.strides[0])], C_reg_18,(2));
__riscv_vse32_v_f32m1(&C.data[(19) * (C.strides[0])], C_reg_19,(2));
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
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
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
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
}

// gemm_RVV_2x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 2] @DRAM
// )
void gemm_RVV_2x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
}

// gemm_RVV_2x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 2] @DRAM
// )
void gemm_RVV_2x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
}

// gemm_RVV_2x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 2] @DRAM
// )
void gemm_RVV_2x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
}

// gemm_RVV_2x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 2] @DRAM
// )
void gemm_RVV_2x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
}

// gemm_RVV_2x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 2] @DRAM
// )
void gemm_RVV_2x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
}

// gemm_RVV_2x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 2] @DRAM
// )
void gemm_RVV_2x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(2));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
}

// gemm_RVV_2x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 2] @DRAM
// )
void gemm_RVV_2x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
}

// gemm_RVV_2x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 2] @DRAM
// )
void gemm_RVV_2x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(2));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(2));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
}

// gemm_RVV_2x7_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 2] @DRAM
// )
void gemm_RVV_2x7_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
}

// gemm_RVV_2x7_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 2] @DRAM
// )
void gemm_RVV_2x7_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(2));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(2));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(2));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
}

// gemm_RVV_2x8_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 2] @DRAM
// )
void gemm_RVV_2x8_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(2));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
}

// gemm_RVV_2x8_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 2] @DRAM
// )
void gemm_RVV_2x8_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(2));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(2));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(2));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(2));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(2));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
}

// gemm_RVV_2x9_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 2] @DRAM
// )
void gemm_RVV_2x9_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_8 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(2));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(2));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
}

// gemm_RVV_2x9_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 2] @DRAM
// )
void gemm_RVV_2x9_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(2));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(2));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(2));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(2));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(2));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(2));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(2));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(2));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(2));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(2));
}

// gemm_RVV_3x10_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 3] @DRAM
// )
void gemm_RVV_3x10_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(3));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(3));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(3));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
}

// gemm_RVV_3x10_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 3] @DRAM
// )
void gemm_RVV_3x10_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(3));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(3));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(3));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(3));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(3));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(3));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(3));
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(3));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(3));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(3));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
}

// gemm_RVV_3x11_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 3] @DRAM
// )
void gemm_RVV_3x11_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(3));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(3));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(3));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(3));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
}

// gemm_RVV_3x11_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 3] @DRAM
// )
void gemm_RVV_3x11_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(3));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(3));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(3));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(3));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(3));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(3));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(3));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(3));
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(3));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(3));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(3));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(3));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
}

// gemm_RVV_3x12_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 3] @DRAM
// )
void gemm_RVV_3x12_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(3));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(3));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(3));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(3));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(3));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
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
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(3));
}

// gemm_RVV_3x12_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 3] @DRAM
// )
void gemm_RVV_3x12_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(3));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(3));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(3));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(3));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(3));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(3));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(3));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(3));
C_reg_11 = __riscv_vle32_v_f32m1(&C.data[(11) * (C.strides[0])],(3));
vfloat32m1_t A_reg;
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(3));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(3));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(3));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(3));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(3));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
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
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(3));
}

// gemm_RVV_3x13_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 3] @DRAM
// )
void gemm_RVV_3x13_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(3));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(3));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(3));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(3));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(3));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(3));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(3));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(3));
}

// gemm_RVV_3x13_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 3] @DRAM
// )
void gemm_RVV_3x13_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(3));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(3));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(3));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(3));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(3));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(3));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(3));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(3));
C_reg_11 = __riscv_vle32_v_f32m1(&C.data[(11) * (C.strides[0])],(3));
C_reg_12 = __riscv_vle32_v_f32m1(&C.data[(12) * (C.strides[0])],(3));
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(3));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(3));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(3));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(3));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(3));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(3));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(3));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(3));
}

// gemm_RVV_3x14_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 3] @DRAM
// )
void gemm_RVV_3x14_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(3));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(3));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(3));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(3));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(3));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(3));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(3));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(3));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(3));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(3));
}

// gemm_RVV_3x14_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 3] @DRAM
// )
void gemm_RVV_3x14_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(3));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(3));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(3));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(3));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(3));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(3));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(3));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(3));
C_reg_11 = __riscv_vle32_v_f32m1(&C.data[(11) * (C.strides[0])],(3));
C_reg_12 = __riscv_vle32_v_f32m1(&C.data[(12) * (C.strides[0])],(3));
C_reg_13 = __riscv_vle32_v_f32m1(&C.data[(13) * (C.strides[0])],(3));
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(3));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(3));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(3));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(3));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(3));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(3));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(3));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(3));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(3));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(3));
}

// gemm_RVV_3x15_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 3] @DRAM
// )
void gemm_RVV_3x15_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(3));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(3));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(3));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(3));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(3));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(3));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(3));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(3));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(3));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(3));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(3));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(3));
}

// gemm_RVV_3x15_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 3] @DRAM
// )
void gemm_RVV_3x15_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(3));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(3));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(3));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(3));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(3));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(3));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(3));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(3));
C_reg_11 = __riscv_vle32_v_f32m1(&C.data[(11) * (C.strides[0])],(3));
C_reg_12 = __riscv_vle32_v_f32m1(&C.data[(12) * (C.strides[0])],(3));
C_reg_13 = __riscv_vle32_v_f32m1(&C.data[(13) * (C.strides[0])],(3));
C_reg_14 = __riscv_vle32_v_f32m1(&C.data[(14) * (C.strides[0])],(3));
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(3));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(3));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(3));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(3));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(3));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(3));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(3));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(3));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(3));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(3));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(3));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(3));
}

// gemm_RVV_3x16_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 3] @DRAM
// )
void gemm_RVV_3x16_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(3));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(3));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(3));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(3));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(3));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(3));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(3));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(3));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(3));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
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
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(3));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(3));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(3));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(3));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(3));
}

// gemm_RVV_3x16_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 3] @DRAM
// )
void gemm_RVV_3x16_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(3));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(3));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(3));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(3));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(3));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(3));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(3));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(3));
C_reg_11 = __riscv_vle32_v_f32m1(&C.data[(11) * (C.strides[0])],(3));
C_reg_12 = __riscv_vle32_v_f32m1(&C.data[(12) * (C.strides[0])],(3));
C_reg_13 = __riscv_vle32_v_f32m1(&C.data[(13) * (C.strides[0])],(3));
C_reg_14 = __riscv_vle32_v_f32m1(&C.data[(14) * (C.strides[0])],(3));
C_reg_15 = __riscv_vle32_v_f32m1(&C.data[(15) * (C.strides[0])],(3));
vfloat32m1_t A_reg;
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(3));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(3));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(3));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(3));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(3));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(3));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(3));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(3));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(3));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
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
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(3));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(3));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(3));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(3));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(3));
}

// gemm_RVV_3x17_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][17, 3] @DRAM
// )
void gemm_RVV_3x17_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t C_reg_16;
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
C_reg_16 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
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
vfloat32m1_t B_reg_16;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(3));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(3));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(3));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(3));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(3));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(3));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(3));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(3));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(3));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(3));
  B_reg_16 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 16],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
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
  C_reg_16 = __riscv_vfmacc_vv_f32m1(C_reg_16, A_reg, B_reg_16,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(3));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(3));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(3));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(3));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(3));
__riscv_vse32_v_f32m1(&C.data[(16) * (C.strides[0])], C_reg_16,(3));
}

// gemm_RVV_3x17_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][17, 3] @DRAM
// )
void gemm_RVV_3x17_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t C_reg_16;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(3));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(3));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(3));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(3));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(3));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(3));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(3));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(3));
C_reg_11 = __riscv_vle32_v_f32m1(&C.data[(11) * (C.strides[0])],(3));
C_reg_12 = __riscv_vle32_v_f32m1(&C.data[(12) * (C.strides[0])],(3));
C_reg_13 = __riscv_vle32_v_f32m1(&C.data[(13) * (C.strides[0])],(3));
C_reg_14 = __riscv_vle32_v_f32m1(&C.data[(14) * (C.strides[0])],(3));
C_reg_15 = __riscv_vle32_v_f32m1(&C.data[(15) * (C.strides[0])],(3));
C_reg_16 = __riscv_vle32_v_f32m1(&C.data[(16) * (C.strides[0])],(3));
vfloat32m1_t A_reg;
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
vfloat32m1_t B_reg_16;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(3));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(3));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(3));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(3));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(3));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(3));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(3));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(3));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(3));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(3));
  B_reg_16 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 16],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
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
  C_reg_16 = __riscv_vfmacc_vv_f32m1(C_reg_16, A_reg, B_reg_16,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(3));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(3));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(3));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(3));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(3));
__riscv_vse32_v_f32m1(&C.data[(16) * (C.strides[0])], C_reg_16,(3));
}

// gemm_RVV_3x18_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][18, 3] @DRAM
// )
void gemm_RVV_3x18_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t C_reg_16;
vfloat32m1_t C_reg_17;
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
C_reg_16 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_17 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
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
vfloat32m1_t B_reg_16;
vfloat32m1_t B_reg_17;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(3));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(3));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(3));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(3));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(3));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(3));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(3));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(3));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(3));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(3));
  B_reg_16 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 16],(3));
  B_reg_17 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 17],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
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
  C_reg_16 = __riscv_vfmacc_vv_f32m1(C_reg_16, A_reg, B_reg_16,(3));
  C_reg_17 = __riscv_vfmacc_vv_f32m1(C_reg_17, A_reg, B_reg_17,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(3));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(3));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(3));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(3));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(3));
__riscv_vse32_v_f32m1(&C.data[(16) * (C.strides[0])], C_reg_16,(3));
__riscv_vse32_v_f32m1(&C.data[(17) * (C.strides[0])], C_reg_17,(3));
}

// gemm_RVV_3x18_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][18, 3] @DRAM
// )
void gemm_RVV_3x18_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t C_reg_16;
vfloat32m1_t C_reg_17;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(3));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(3));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(3));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(3));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(3));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(3));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(3));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(3));
C_reg_11 = __riscv_vle32_v_f32m1(&C.data[(11) * (C.strides[0])],(3));
C_reg_12 = __riscv_vle32_v_f32m1(&C.data[(12) * (C.strides[0])],(3));
C_reg_13 = __riscv_vle32_v_f32m1(&C.data[(13) * (C.strides[0])],(3));
C_reg_14 = __riscv_vle32_v_f32m1(&C.data[(14) * (C.strides[0])],(3));
C_reg_15 = __riscv_vle32_v_f32m1(&C.data[(15) * (C.strides[0])],(3));
C_reg_16 = __riscv_vle32_v_f32m1(&C.data[(16) * (C.strides[0])],(3));
C_reg_17 = __riscv_vle32_v_f32m1(&C.data[(17) * (C.strides[0])],(3));
vfloat32m1_t A_reg;
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
vfloat32m1_t B_reg_16;
vfloat32m1_t B_reg_17;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(3));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(3));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(3));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(3));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(3));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(3));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(3));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(3));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(3));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(3));
  B_reg_16 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 16],(3));
  B_reg_17 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 17],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
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
  C_reg_16 = __riscv_vfmacc_vv_f32m1(C_reg_16, A_reg, B_reg_16,(3));
  C_reg_17 = __riscv_vfmacc_vv_f32m1(C_reg_17, A_reg, B_reg_17,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(3));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(3));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(3));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(3));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(3));
__riscv_vse32_v_f32m1(&C.data[(16) * (C.strides[0])], C_reg_16,(3));
__riscv_vse32_v_f32m1(&C.data[(17) * (C.strides[0])], C_reg_17,(3));
}

// gemm_RVV_3x19_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][19, 3] @DRAM
// )
void gemm_RVV_3x19_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t C_reg_16;
vfloat32m1_t C_reg_17;
vfloat32m1_t C_reg_18;
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
C_reg_16 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_17 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_18 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
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
vfloat32m1_t B_reg_16;
vfloat32m1_t B_reg_17;
vfloat32m1_t B_reg_18;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(3));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(3));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(3));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(3));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(3));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(3));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(3));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(3));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(3));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(3));
  B_reg_16 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 16],(3));
  B_reg_17 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 17],(3));
  B_reg_18 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 18],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
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
  C_reg_16 = __riscv_vfmacc_vv_f32m1(C_reg_16, A_reg, B_reg_16,(3));
  C_reg_17 = __riscv_vfmacc_vv_f32m1(C_reg_17, A_reg, B_reg_17,(3));
  C_reg_18 = __riscv_vfmacc_vv_f32m1(C_reg_18, A_reg, B_reg_18,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(3));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(3));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(3));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(3));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(3));
__riscv_vse32_v_f32m1(&C.data[(16) * (C.strides[0])], C_reg_16,(3));
__riscv_vse32_v_f32m1(&C.data[(17) * (C.strides[0])], C_reg_17,(3));
__riscv_vse32_v_f32m1(&C.data[(18) * (C.strides[0])], C_reg_18,(3));
}

// gemm_RVV_3x19_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][19, 3] @DRAM
// )
void gemm_RVV_3x19_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t C_reg_16;
vfloat32m1_t C_reg_17;
vfloat32m1_t C_reg_18;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(3));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(3));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(3));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(3));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(3));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(3));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(3));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(3));
C_reg_11 = __riscv_vle32_v_f32m1(&C.data[(11) * (C.strides[0])],(3));
C_reg_12 = __riscv_vle32_v_f32m1(&C.data[(12) * (C.strides[0])],(3));
C_reg_13 = __riscv_vle32_v_f32m1(&C.data[(13) * (C.strides[0])],(3));
C_reg_14 = __riscv_vle32_v_f32m1(&C.data[(14) * (C.strides[0])],(3));
C_reg_15 = __riscv_vle32_v_f32m1(&C.data[(15) * (C.strides[0])],(3));
C_reg_16 = __riscv_vle32_v_f32m1(&C.data[(16) * (C.strides[0])],(3));
C_reg_17 = __riscv_vle32_v_f32m1(&C.data[(17) * (C.strides[0])],(3));
C_reg_18 = __riscv_vle32_v_f32m1(&C.data[(18) * (C.strides[0])],(3));
vfloat32m1_t A_reg;
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
vfloat32m1_t B_reg_16;
vfloat32m1_t B_reg_17;
vfloat32m1_t B_reg_18;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(3));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(3));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(3));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(3));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(3));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(3));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(3));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(3));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(3));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(3));
  B_reg_16 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 16],(3));
  B_reg_17 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 17],(3));
  B_reg_18 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 18],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
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
  C_reg_16 = __riscv_vfmacc_vv_f32m1(C_reg_16, A_reg, B_reg_16,(3));
  C_reg_17 = __riscv_vfmacc_vv_f32m1(C_reg_17, A_reg, B_reg_17,(3));
  C_reg_18 = __riscv_vfmacc_vv_f32m1(C_reg_18, A_reg, B_reg_18,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(3));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(3));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(3));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(3));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(3));
__riscv_vse32_v_f32m1(&C.data[(16) * (C.strides[0])], C_reg_16,(3));
__riscv_vse32_v_f32m1(&C.data[(17) * (C.strides[0])], C_reg_17,(3));
__riscv_vse32_v_f32m1(&C.data[(18) * (C.strides[0])], C_reg_18,(3));
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
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
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
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
}

// gemm_RVV_3x20_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 3] @DRAM
// )
void gemm_RVV_3x20_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t C_reg_16;
vfloat32m1_t C_reg_17;
vfloat32m1_t C_reg_18;
vfloat32m1_t C_reg_19;
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
C_reg_16 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_17 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_18 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_19 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
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
vfloat32m1_t B_reg_16;
vfloat32m1_t B_reg_17;
vfloat32m1_t B_reg_18;
vfloat32m1_t B_reg_19;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(3));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(3));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(3));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(3));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(3));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(3));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(3));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(3));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(3));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(3));
  B_reg_16 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 16],(3));
  B_reg_17 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 17],(3));
  B_reg_18 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 18],(3));
  B_reg_19 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 19],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
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
  C_reg_16 = __riscv_vfmacc_vv_f32m1(C_reg_16, A_reg, B_reg_16,(3));
  C_reg_17 = __riscv_vfmacc_vv_f32m1(C_reg_17, A_reg, B_reg_17,(3));
  C_reg_18 = __riscv_vfmacc_vv_f32m1(C_reg_18, A_reg, B_reg_18,(3));
  C_reg_19 = __riscv_vfmacc_vv_f32m1(C_reg_19, A_reg, B_reg_19,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(3));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(3));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(3));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(3));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(3));
__riscv_vse32_v_f32m1(&C.data[(16) * (C.strides[0])], C_reg_16,(3));
__riscv_vse32_v_f32m1(&C.data[(17) * (C.strides[0])], C_reg_17,(3));
__riscv_vse32_v_f32m1(&C.data[(18) * (C.strides[0])], C_reg_18,(3));
__riscv_vse32_v_f32m1(&C.data[(19) * (C.strides[0])], C_reg_19,(3));
}

// gemm_RVV_3x20_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 3] @DRAM
// )
void gemm_RVV_3x20_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t C_reg_16;
vfloat32m1_t C_reg_17;
vfloat32m1_t C_reg_18;
vfloat32m1_t C_reg_19;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(3));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(3));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(3));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(3));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(3));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(3));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(3));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(3));
C_reg_11 = __riscv_vle32_v_f32m1(&C.data[(11) * (C.strides[0])],(3));
C_reg_12 = __riscv_vle32_v_f32m1(&C.data[(12) * (C.strides[0])],(3));
C_reg_13 = __riscv_vle32_v_f32m1(&C.data[(13) * (C.strides[0])],(3));
C_reg_14 = __riscv_vle32_v_f32m1(&C.data[(14) * (C.strides[0])],(3));
C_reg_15 = __riscv_vle32_v_f32m1(&C.data[(15) * (C.strides[0])],(3));
C_reg_16 = __riscv_vle32_v_f32m1(&C.data[(16) * (C.strides[0])],(3));
C_reg_17 = __riscv_vle32_v_f32m1(&C.data[(17) * (C.strides[0])],(3));
C_reg_18 = __riscv_vle32_v_f32m1(&C.data[(18) * (C.strides[0])],(3));
C_reg_19 = __riscv_vle32_v_f32m1(&C.data[(19) * (C.strides[0])],(3));
vfloat32m1_t A_reg;
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
vfloat32m1_t B_reg_16;
vfloat32m1_t B_reg_17;
vfloat32m1_t B_reg_18;
vfloat32m1_t B_reg_19;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(3));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(3));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(3));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(3));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(3));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(3));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(3));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(3));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(3));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(3));
  B_reg_16 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 16],(3));
  B_reg_17 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 17],(3));
  B_reg_18 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 18],(3));
  B_reg_19 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 19],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
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
  C_reg_16 = __riscv_vfmacc_vv_f32m1(C_reg_16, A_reg, B_reg_16,(3));
  C_reg_17 = __riscv_vfmacc_vv_f32m1(C_reg_17, A_reg, B_reg_17,(3));
  C_reg_18 = __riscv_vfmacc_vv_f32m1(C_reg_18, A_reg, B_reg_18,(3));
  C_reg_19 = __riscv_vfmacc_vv_f32m1(C_reg_19, A_reg, B_reg_19,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(3));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(3));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(3));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(3));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(3));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(3));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(3));
__riscv_vse32_v_f32m1(&C.data[(16) * (C.strides[0])], C_reg_16,(3));
__riscv_vse32_v_f32m1(&C.data[(17) * (C.strides[0])], C_reg_17,(3));
__riscv_vse32_v_f32m1(&C.data[(18) * (C.strides[0])], C_reg_18,(3));
__riscv_vse32_v_f32m1(&C.data[(19) * (C.strides[0])], C_reg_19,(3));
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
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
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
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
}

// gemm_RVV_3x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 3] @DRAM
// )
void gemm_RVV_3x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
}

// gemm_RVV_3x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 3] @DRAM
// )
void gemm_RVV_3x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
}

// gemm_RVV_3x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 3] @DRAM
// )
void gemm_RVV_3x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
}

// gemm_RVV_3x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 3] @DRAM
// )
void gemm_RVV_3x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
}

// gemm_RVV_3x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 3] @DRAM
// )
void gemm_RVV_3x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
}

// gemm_RVV_3x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 3] @DRAM
// )
void gemm_RVV_3x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(3));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
}

// gemm_RVV_3x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 3] @DRAM
// )
void gemm_RVV_3x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
}

// gemm_RVV_3x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 3] @DRAM
// )
void gemm_RVV_3x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(3));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(3));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
}

// gemm_RVV_3x7_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 3] @DRAM
// )
void gemm_RVV_3x7_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
}

// gemm_RVV_3x7_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 3] @DRAM
// )
void gemm_RVV_3x7_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(3));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(3));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(3));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
}

// gemm_RVV_3x8_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 3] @DRAM
// )
void gemm_RVV_3x8_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(3));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
}

// gemm_RVV_3x8_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 3] @DRAM
// )
void gemm_RVV_3x8_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(3));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(3));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(3));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(3));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(3));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
}

// gemm_RVV_3x9_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 3] @DRAM
// )
void gemm_RVV_3x9_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_8 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(3));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(3));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
}

// gemm_RVV_3x9_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 3] @DRAM
// )
void gemm_RVV_3x9_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(3));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(3));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(3));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(3));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(3));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(3));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(3));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(3));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(3));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(3));
}

// gemm_RVV_4x10_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 4] @DRAM
// )
void gemm_RVV_4x10_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(4));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
}

// gemm_RVV_4x10_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 4] @DRAM
// )
void gemm_RVV_4x10_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(4));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(4));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(4));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(4));
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(4));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
}

// gemm_RVV_4x11_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 4] @DRAM
// )
void gemm_RVV_4x11_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(4));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(4));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
}

// gemm_RVV_4x11_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 4] @DRAM
// )
void gemm_RVV_4x11_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(4));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(4));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(4));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(4));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(4));
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(4));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(4));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
}

// gemm_RVV_4x12_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 4] @DRAM
// )
void gemm_RVV_4x12_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(4));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(4));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(4));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
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
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(4));
}

// gemm_RVV_4x12_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 4] @DRAM
// )
void gemm_RVV_4x12_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(4));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(4));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(4));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(4));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(4));
C_reg_11 = __riscv_vle32_v_f32m1(&C.data[(11) * (C.strides[0])],(4));
vfloat32m1_t A_reg;
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(4));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(4));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(4));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
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
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(4));
}

// gemm_RVV_4x13_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 4] @DRAM
// )
void gemm_RVV_4x13_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(4));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(4));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(4));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(4));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(4));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(4));
}

// gemm_RVV_4x13_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 4] @DRAM
// )
void gemm_RVV_4x13_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(4));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(4));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(4));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(4));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(4));
C_reg_11 = __riscv_vle32_v_f32m1(&C.data[(11) * (C.strides[0])],(4));
C_reg_12 = __riscv_vle32_v_f32m1(&C.data[(12) * (C.strides[0])],(4));
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(4));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(4));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(4));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(4));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(4));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(4));
}

// gemm_RVV_4x14_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 4] @DRAM
// )
void gemm_RVV_4x14_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(4));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(4));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(4));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(4));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(4));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(4));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(4));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(4));
}

// gemm_RVV_4x14_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 4] @DRAM
// )
void gemm_RVV_4x14_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(4));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(4));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(4));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(4));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(4));
C_reg_11 = __riscv_vle32_v_f32m1(&C.data[(11) * (C.strides[0])],(4));
C_reg_12 = __riscv_vle32_v_f32m1(&C.data[(12) * (C.strides[0])],(4));
C_reg_13 = __riscv_vle32_v_f32m1(&C.data[(13) * (C.strides[0])],(4));
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(4));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(4));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(4));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(4));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(4));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(4));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(4));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(4));
}

// gemm_RVV_4x15_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 4] @DRAM
// )
void gemm_RVV_4x15_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(4));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(4));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(4));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(4));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(4));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(4));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(4));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(4));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(4));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(4));
}

// gemm_RVV_4x15_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 4] @DRAM
// )
void gemm_RVV_4x15_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(4));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(4));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(4));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(4));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(4));
C_reg_11 = __riscv_vle32_v_f32m1(&C.data[(11) * (C.strides[0])],(4));
C_reg_12 = __riscv_vle32_v_f32m1(&C.data[(12) * (C.strides[0])],(4));
C_reg_13 = __riscv_vle32_v_f32m1(&C.data[(13) * (C.strides[0])],(4));
C_reg_14 = __riscv_vle32_v_f32m1(&C.data[(14) * (C.strides[0])],(4));
vfloat32m1_t A_reg;
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
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(4));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(4));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(4));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(4));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(4));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(4));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
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
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(4));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(4));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(4));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(4));
}

// gemm_RVV_4x16_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 4] @DRAM
// )
void gemm_RVV_4x16_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(4));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(4));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(4));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(4));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(4));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(4));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(4));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
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
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(4));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(4));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(4));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(4));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(4));
}

// gemm_RVV_4x16_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 4] @DRAM
// )
void gemm_RVV_4x16_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(4));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(4));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(4));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(4));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(4));
C_reg_11 = __riscv_vle32_v_f32m1(&C.data[(11) * (C.strides[0])],(4));
C_reg_12 = __riscv_vle32_v_f32m1(&C.data[(12) * (C.strides[0])],(4));
C_reg_13 = __riscv_vle32_v_f32m1(&C.data[(13) * (C.strides[0])],(4));
C_reg_14 = __riscv_vle32_v_f32m1(&C.data[(14) * (C.strides[0])],(4));
C_reg_15 = __riscv_vle32_v_f32m1(&C.data[(15) * (C.strides[0])],(4));
vfloat32m1_t A_reg;
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(4));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(4));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(4));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(4));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(4));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(4));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(4));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
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
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(4));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(4));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(4));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(4));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(4));
}

// gemm_RVV_4x17_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][17, 4] @DRAM
// )
void gemm_RVV_4x17_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t C_reg_16;
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
C_reg_16 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
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
vfloat32m1_t B_reg_16;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(4));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(4));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(4));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(4));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(4));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(4));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(4));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(4));
  B_reg_16 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 16],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
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
  C_reg_16 = __riscv_vfmacc_vv_f32m1(C_reg_16, A_reg, B_reg_16,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(4));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(4));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(4));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(4));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(4));
__riscv_vse32_v_f32m1(&C.data[(16) * (C.strides[0])], C_reg_16,(4));
}

// gemm_RVV_4x17_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][17, 4] @DRAM
// )
void gemm_RVV_4x17_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t C_reg_16;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(4));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(4));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(4));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(4));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(4));
C_reg_11 = __riscv_vle32_v_f32m1(&C.data[(11) * (C.strides[0])],(4));
C_reg_12 = __riscv_vle32_v_f32m1(&C.data[(12) * (C.strides[0])],(4));
C_reg_13 = __riscv_vle32_v_f32m1(&C.data[(13) * (C.strides[0])],(4));
C_reg_14 = __riscv_vle32_v_f32m1(&C.data[(14) * (C.strides[0])],(4));
C_reg_15 = __riscv_vle32_v_f32m1(&C.data[(15) * (C.strides[0])],(4));
C_reg_16 = __riscv_vle32_v_f32m1(&C.data[(16) * (C.strides[0])],(4));
vfloat32m1_t A_reg;
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
vfloat32m1_t B_reg_16;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(4));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(4));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(4));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(4));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(4));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(4));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(4));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(4));
  B_reg_16 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 16],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
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
  C_reg_16 = __riscv_vfmacc_vv_f32m1(C_reg_16, A_reg, B_reg_16,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(4));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(4));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(4));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(4));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(4));
__riscv_vse32_v_f32m1(&C.data[(16) * (C.strides[0])], C_reg_16,(4));
}

// gemm_RVV_4x18_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][18, 4] @DRAM
// )
void gemm_RVV_4x18_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t C_reg_16;
vfloat32m1_t C_reg_17;
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
C_reg_16 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_17 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
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
vfloat32m1_t B_reg_16;
vfloat32m1_t B_reg_17;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(4));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(4));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(4));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(4));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(4));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(4));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(4));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(4));
  B_reg_16 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 16],(4));
  B_reg_17 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 17],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
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
  C_reg_16 = __riscv_vfmacc_vv_f32m1(C_reg_16, A_reg, B_reg_16,(4));
  C_reg_17 = __riscv_vfmacc_vv_f32m1(C_reg_17, A_reg, B_reg_17,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(4));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(4));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(4));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(4));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(4));
__riscv_vse32_v_f32m1(&C.data[(16) * (C.strides[0])], C_reg_16,(4));
__riscv_vse32_v_f32m1(&C.data[(17) * (C.strides[0])], C_reg_17,(4));
}

// gemm_RVV_4x18_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][18, 4] @DRAM
// )
void gemm_RVV_4x18_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t C_reg_16;
vfloat32m1_t C_reg_17;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(4));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(4));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(4));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(4));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(4));
C_reg_11 = __riscv_vle32_v_f32m1(&C.data[(11) * (C.strides[0])],(4));
C_reg_12 = __riscv_vle32_v_f32m1(&C.data[(12) * (C.strides[0])],(4));
C_reg_13 = __riscv_vle32_v_f32m1(&C.data[(13) * (C.strides[0])],(4));
C_reg_14 = __riscv_vle32_v_f32m1(&C.data[(14) * (C.strides[0])],(4));
C_reg_15 = __riscv_vle32_v_f32m1(&C.data[(15) * (C.strides[0])],(4));
C_reg_16 = __riscv_vle32_v_f32m1(&C.data[(16) * (C.strides[0])],(4));
C_reg_17 = __riscv_vle32_v_f32m1(&C.data[(17) * (C.strides[0])],(4));
vfloat32m1_t A_reg;
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
vfloat32m1_t B_reg_16;
vfloat32m1_t B_reg_17;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(4));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(4));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(4));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(4));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(4));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(4));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(4));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(4));
  B_reg_16 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 16],(4));
  B_reg_17 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 17],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
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
  C_reg_16 = __riscv_vfmacc_vv_f32m1(C_reg_16, A_reg, B_reg_16,(4));
  C_reg_17 = __riscv_vfmacc_vv_f32m1(C_reg_17, A_reg, B_reg_17,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(4));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(4));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(4));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(4));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(4));
__riscv_vse32_v_f32m1(&C.data[(16) * (C.strides[0])], C_reg_16,(4));
__riscv_vse32_v_f32m1(&C.data[(17) * (C.strides[0])], C_reg_17,(4));
}

// gemm_RVV_4x19_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][19, 4] @DRAM
// )
void gemm_RVV_4x19_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t C_reg_16;
vfloat32m1_t C_reg_17;
vfloat32m1_t C_reg_18;
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
C_reg_16 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_17 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_18 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
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
vfloat32m1_t B_reg_16;
vfloat32m1_t B_reg_17;
vfloat32m1_t B_reg_18;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(4));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(4));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(4));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(4));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(4));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(4));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(4));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(4));
  B_reg_16 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 16],(4));
  B_reg_17 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 17],(4));
  B_reg_18 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 18],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
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
  C_reg_16 = __riscv_vfmacc_vv_f32m1(C_reg_16, A_reg, B_reg_16,(4));
  C_reg_17 = __riscv_vfmacc_vv_f32m1(C_reg_17, A_reg, B_reg_17,(4));
  C_reg_18 = __riscv_vfmacc_vv_f32m1(C_reg_18, A_reg, B_reg_18,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(4));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(4));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(4));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(4));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(4));
__riscv_vse32_v_f32m1(&C.data[(16) * (C.strides[0])], C_reg_16,(4));
__riscv_vse32_v_f32m1(&C.data[(17) * (C.strides[0])], C_reg_17,(4));
__riscv_vse32_v_f32m1(&C.data[(18) * (C.strides[0])], C_reg_18,(4));
}

// gemm_RVV_4x19_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][19, 4] @DRAM
// )
void gemm_RVV_4x19_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t C_reg_16;
vfloat32m1_t C_reg_17;
vfloat32m1_t C_reg_18;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(4));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(4));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(4));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(4));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(4));
C_reg_11 = __riscv_vle32_v_f32m1(&C.data[(11) * (C.strides[0])],(4));
C_reg_12 = __riscv_vle32_v_f32m1(&C.data[(12) * (C.strides[0])],(4));
C_reg_13 = __riscv_vle32_v_f32m1(&C.data[(13) * (C.strides[0])],(4));
C_reg_14 = __riscv_vle32_v_f32m1(&C.data[(14) * (C.strides[0])],(4));
C_reg_15 = __riscv_vle32_v_f32m1(&C.data[(15) * (C.strides[0])],(4));
C_reg_16 = __riscv_vle32_v_f32m1(&C.data[(16) * (C.strides[0])],(4));
C_reg_17 = __riscv_vle32_v_f32m1(&C.data[(17) * (C.strides[0])],(4));
C_reg_18 = __riscv_vle32_v_f32m1(&C.data[(18) * (C.strides[0])],(4));
vfloat32m1_t A_reg;
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
vfloat32m1_t B_reg_16;
vfloat32m1_t B_reg_17;
vfloat32m1_t B_reg_18;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(4));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(4));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(4));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(4));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(4));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(4));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(4));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(4));
  B_reg_16 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 16],(4));
  B_reg_17 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 17],(4));
  B_reg_18 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 18],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
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
  C_reg_16 = __riscv_vfmacc_vv_f32m1(C_reg_16, A_reg, B_reg_16,(4));
  C_reg_17 = __riscv_vfmacc_vv_f32m1(C_reg_17, A_reg, B_reg_17,(4));
  C_reg_18 = __riscv_vfmacc_vv_f32m1(C_reg_18, A_reg, B_reg_18,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(4));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(4));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(4));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(4));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(4));
__riscv_vse32_v_f32m1(&C.data[(16) * (C.strides[0])], C_reg_16,(4));
__riscv_vse32_v_f32m1(&C.data[(17) * (C.strides[0])], C_reg_17,(4));
__riscv_vse32_v_f32m1(&C.data[(18) * (C.strides[0])], C_reg_18,(4));
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
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
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
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
}

// gemm_RVV_4x20_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 4] @DRAM
// )
void gemm_RVV_4x20_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t C_reg_16;
vfloat32m1_t C_reg_17;
vfloat32m1_t C_reg_18;
vfloat32m1_t C_reg_19;
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
C_reg_16 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_17 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_18 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_19 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
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
vfloat32m1_t B_reg_16;
vfloat32m1_t B_reg_17;
vfloat32m1_t B_reg_18;
vfloat32m1_t B_reg_19;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(4));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(4));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(4));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(4));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(4));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(4));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(4));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(4));
  B_reg_16 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 16],(4));
  B_reg_17 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 17],(4));
  B_reg_18 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 18],(4));
  B_reg_19 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 19],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
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
  C_reg_16 = __riscv_vfmacc_vv_f32m1(C_reg_16, A_reg, B_reg_16,(4));
  C_reg_17 = __riscv_vfmacc_vv_f32m1(C_reg_17, A_reg, B_reg_17,(4));
  C_reg_18 = __riscv_vfmacc_vv_f32m1(C_reg_18, A_reg, B_reg_18,(4));
  C_reg_19 = __riscv_vfmacc_vv_f32m1(C_reg_19, A_reg, B_reg_19,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(4));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(4));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(4));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(4));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(4));
__riscv_vse32_v_f32m1(&C.data[(16) * (C.strides[0])], C_reg_16,(4));
__riscv_vse32_v_f32m1(&C.data[(17) * (C.strides[0])], C_reg_17,(4));
__riscv_vse32_v_f32m1(&C.data[(18) * (C.strides[0])], C_reg_18,(4));
__riscv_vse32_v_f32m1(&C.data[(19) * (C.strides[0])], C_reg_19,(4));
}

// gemm_RVV_4x20_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 4] @DRAM
// )
void gemm_RVV_4x20_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t C_reg_16;
vfloat32m1_t C_reg_17;
vfloat32m1_t C_reg_18;
vfloat32m1_t C_reg_19;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(4));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(4));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(4));
C_reg_9 = __riscv_vle32_v_f32m1(&C.data[(9) * (C.strides[0])],(4));
C_reg_10 = __riscv_vle32_v_f32m1(&C.data[(10) * (C.strides[0])],(4));
C_reg_11 = __riscv_vle32_v_f32m1(&C.data[(11) * (C.strides[0])],(4));
C_reg_12 = __riscv_vle32_v_f32m1(&C.data[(12) * (C.strides[0])],(4));
C_reg_13 = __riscv_vle32_v_f32m1(&C.data[(13) * (C.strides[0])],(4));
C_reg_14 = __riscv_vle32_v_f32m1(&C.data[(14) * (C.strides[0])],(4));
C_reg_15 = __riscv_vle32_v_f32m1(&C.data[(15) * (C.strides[0])],(4));
C_reg_16 = __riscv_vle32_v_f32m1(&C.data[(16) * (C.strides[0])],(4));
C_reg_17 = __riscv_vle32_v_f32m1(&C.data[(17) * (C.strides[0])],(4));
C_reg_18 = __riscv_vle32_v_f32m1(&C.data[(18) * (C.strides[0])],(4));
C_reg_19 = __riscv_vle32_v_f32m1(&C.data[(19) * (C.strides[0])],(4));
vfloat32m1_t A_reg;
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
vfloat32m1_t B_reg_16;
vfloat32m1_t B_reg_17;
vfloat32m1_t B_reg_18;
vfloat32m1_t B_reg_19;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(4));
  B_reg_9 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 9],(4));
  B_reg_10 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 10],(4));
  B_reg_11 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 11],(4));
  B_reg_12 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 12],(4));
  B_reg_13 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 13],(4));
  B_reg_14 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 14],(4));
  B_reg_15 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 15],(4));
  B_reg_16 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 16],(4));
  B_reg_17 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 17],(4));
  B_reg_18 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 18],(4));
  B_reg_19 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 19],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
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
  C_reg_16 = __riscv_vfmacc_vv_f32m1(C_reg_16, A_reg, B_reg_16,(4));
  C_reg_17 = __riscv_vfmacc_vv_f32m1(C_reg_17, A_reg, B_reg_17,(4));
  C_reg_18 = __riscv_vfmacc_vv_f32m1(C_reg_18, A_reg, B_reg_18,(4));
  C_reg_19 = __riscv_vfmacc_vv_f32m1(C_reg_19, A_reg, B_reg_19,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
__riscv_vse32_v_f32m1(&C.data[(9) * (C.strides[0])], C_reg_9,(4));
__riscv_vse32_v_f32m1(&C.data[(10) * (C.strides[0])], C_reg_10,(4));
__riscv_vse32_v_f32m1(&C.data[(11) * (C.strides[0])], C_reg_11,(4));
__riscv_vse32_v_f32m1(&C.data[(12) * (C.strides[0])], C_reg_12,(4));
__riscv_vse32_v_f32m1(&C.data[(13) * (C.strides[0])], C_reg_13,(4));
__riscv_vse32_v_f32m1(&C.data[(14) * (C.strides[0])], C_reg_14,(4));
__riscv_vse32_v_f32m1(&C.data[(15) * (C.strides[0])], C_reg_15,(4));
__riscv_vse32_v_f32m1(&C.data[(16) * (C.strides[0])], C_reg_16,(4));
__riscv_vse32_v_f32m1(&C.data[(17) * (C.strides[0])], C_reg_17,(4));
__riscv_vse32_v_f32m1(&C.data[(18) * (C.strides[0])], C_reg_18,(4));
__riscv_vse32_v_f32m1(&C.data[(19) * (C.strides[0])], C_reg_19,(4));
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
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
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
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
}

// gemm_RVV_4x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 4] @DRAM
// )
void gemm_RVV_4x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
}

// gemm_RVV_4x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 4] @DRAM
// )
void gemm_RVV_4x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
}

// gemm_RVV_4x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 4] @DRAM
// )
void gemm_RVV_4x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
}

// gemm_RVV_4x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 4] @DRAM
// )
void gemm_RVV_4x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
}

// gemm_RVV_4x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 4] @DRAM
// )
void gemm_RVV_4x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
}

// gemm_RVV_4x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 4] @DRAM
// )
void gemm_RVV_4x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
}

// gemm_RVV_4x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 4] @DRAM
// )
void gemm_RVV_4x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
}

// gemm_RVV_4x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 4] @DRAM
// )
void gemm_RVV_4x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
}

// gemm_RVV_4x7_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 4] @DRAM
// )
void gemm_RVV_4x7_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
}

// gemm_RVV_4x7_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 4] @DRAM
// )
void gemm_RVV_4x7_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
}

// gemm_RVV_4x8_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 4] @DRAM
// )
void gemm_RVV_4x8_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
}

// gemm_RVV_4x8_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 4] @DRAM
// )
void gemm_RVV_4x8_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(4));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
}

// gemm_RVV_4x9_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 4] @DRAM
// )
void gemm_RVV_4x9_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_8 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
}

// gemm_RVV_4x9_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 4] @DRAM
// )
void gemm_RVV_4x9_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C ) {
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
C_reg_0 = __riscv_vle32_v_f32m1(&C.data[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C.data[C.strides[0]],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C.data[(2) * (C.strides[0])],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C.data[(3) * (C.strides[0])],(4));
C_reg_4 = __riscv_vle32_v_f32m1(&C.data[(4) * (C.strides[0])],(4));
C_reg_5 = __riscv_vle32_v_f32m1(&C.data[(5) * (C.strides[0])],(4));
C_reg_6 = __riscv_vle32_v_f32m1(&C.data[(6) * (C.strides[0])],(4));
C_reg_7 = __riscv_vle32_v_f32m1(&C.data[(7) * (C.strides[0])],(4));
C_reg_8 = __riscv_vle32_v_f32m1(&C.data[(8) * (C.strides[0])],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
for (int_fast32_t k = 0; k < KC; k++) {
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0])],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 6],(4));
  B_reg_7 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 7],(4));
  B_reg_8 = __riscv_vfmv_v_f_f32m1(B.data[(k) * (B.strides[0]) + 8],(4));
  A_reg = __riscv_vle32_v_f32m1(&A.data[(k) * (A.strides[0])],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(4));
}
__riscv_vse32_v_f32m1(&C.data[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C.data[C.strides[0]], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C.data[(2) * (C.strides[0])], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C.data[(3) * (C.strides[0])], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C.data[(4) * (C.strides[0])], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C.data[(5) * (C.strides[0])], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C.data[(6) * (C.strides[0])], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C.data[(7) * (C.strides[0])], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C.data[(8) * (C.strides[0])], C_reg_8,(4));
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