#include "exo_matrix_RVV_fp32.h"

ukrFunction**** allocateMatrix() {
    ukrFunction**** matrix = (ukrFunction****)malloc(9 * sizeof(ukrFunction***));
    for (int i = 0; i < 9; i++) {
        matrix[i] = (ukrFunction***)malloc(25 * sizeof(ukrFunction**));
        for (int j = 0; j < 25; j++) {
            matrix[i][j] = (ukrFunction**)malloc(2 * sizeof(ukrFunction*));
            for (int b = 0; b < 2; b++) {
                matrix[i][j][b] = (ukrFunction*)malloc(1 * sizeof(ukrFunction));
            }
        }
    }
    return matrix;
}


void fillMatrix(ukrFunction**** matrix) {
    *matrix[0][0][0] = 	(ukrFunction)NULL;
    *matrix[0][0][1] = 	(ukrFunction)NULL;
    *matrix[0][1][0] = 	(ukrFunction)NULL;
    *matrix[0][1][1] = 	(ukrFunction)NULL;
    *matrix[0][2][0] = 	(ukrFunction)NULL;
    *matrix[0][2][1] = 	(ukrFunction)NULL;
    *matrix[0][3][0] = 	(ukrFunction)NULL;
    *matrix[0][3][1] = 	(ukrFunction)NULL;
    *matrix[0][4][0] = 	(ukrFunction)NULL;
    *matrix[0][4][1] = 	(ukrFunction)NULL;
    *matrix[0][5][0] = 	(ukrFunction)NULL;
    *matrix[0][5][1] = 	(ukrFunction)NULL;
    *matrix[0][6][0] = 	(ukrFunction)NULL;
    *matrix[0][6][1] = 	(ukrFunction)NULL;
    *matrix[0][7][0] = 	(ukrFunction)NULL;
    *matrix[0][7][1] = 	(ukrFunction)NULL;
    *matrix[0][8][0] = 	(ukrFunction)NULL;
    *matrix[0][8][1] = 	(ukrFunction)NULL;
    *matrix[0][9][0] = 	(ukrFunction)NULL;
    *matrix[0][9][1] = 	(ukrFunction)NULL;
    *matrix[0][10][0] = 	(ukrFunction)NULL;
    *matrix[0][10][1] = 	(ukrFunction)NULL;
    *matrix[0][11][0] = 	(ukrFunction)NULL;
    *matrix[0][11][1] = 	(ukrFunction)NULL;
    *matrix[0][12][0] = 	(ukrFunction)NULL;
    *matrix[0][12][1] = 	(ukrFunction)NULL;
    *matrix[0][13][0] = 	(ukrFunction)NULL;
    *matrix[0][13][1] = 	(ukrFunction)NULL;
    *matrix[0][14][0] = 	(ukrFunction)NULL;
    *matrix[0][14][1] = 	(ukrFunction)NULL;
    *matrix[0][15][0] = 	(ukrFunction)NULL;
    *matrix[0][15][1] = 	(ukrFunction)NULL;
    *matrix[0][16][0] = 	(ukrFunction)NULL;
    *matrix[0][16][1] = 	(ukrFunction)NULL;
    *matrix[0][17][0] = 	(ukrFunction)NULL;
    *matrix[0][17][1] = 	(ukrFunction)NULL;
    *matrix[0][18][0] = 	(ukrFunction)NULL;
    *matrix[0][18][1] = 	(ukrFunction)NULL;
    *matrix[0][19][0] = 	(ukrFunction)NULL;
    *matrix[0][19][1] = 	(ukrFunction)NULL;
    *matrix[0][20][0] = 	(ukrFunction)NULL;
    *matrix[0][20][1] = 	(ukrFunction)NULL;
    *matrix[0][21][0] = 	(ukrFunction)NULL;
    *matrix[0][21][1] = 	(ukrFunction)NULL;
    *matrix[0][22][0] = 	(ukrFunction)NULL;
    *matrix[0][22][1] = 	(ukrFunction)NULL;
    *matrix[0][23][0] = 	(ukrFunction)NULL;
    *matrix[0][23][1] = 	(ukrFunction)NULL;
    *matrix[0][24][0] = 	(ukrFunction)NULL;
    *matrix[0][24][1] = 	(ukrFunction)NULL;
    *matrix[1][0][0] = 	(ukrFunction)NULL;
    *matrix[1][0][1] = 	(ukrFunction)NULL;
    *matrix[1][1][0] = 	(ukrFunction)gemm_RISCV_1x1_b0_col_fp32;
    *matrix[1][1][1] = 	(ukrFunction)gemm_RISCV_1x1_b1_col_fp32;
    *matrix[1][2][0] = 	(ukrFunction)gemm_RISCV_1x2_b0_col_fp32;
    *matrix[1][2][1] = 	(ukrFunction)gemm_RISCV_1x2_b1_col_fp32;
    *matrix[1][3][0] = 	(ukrFunction)gemm_RISCV_1x3_b0_col_fp32;
    *matrix[1][3][1] = 	(ukrFunction)gemm_RISCV_1x3_b1_col_fp32;
    *matrix[1][4][0] = 	(ukrFunction)gemm_RISCV_1x4_b0_col_fp32;
    *matrix[1][4][1] = 	(ukrFunction)gemm_RISCV_1x4_b1_col_fp32;
    *matrix[1][5][0] = 	(ukrFunction)gemm_RISCV_1x5_b0_col_fp32;
    *matrix[1][5][1] = 	(ukrFunction)gemm_RISCV_1x5_b1_col_fp32;
    *matrix[1][6][0] = 	(ukrFunction)gemm_RISCV_1x6_b0_col_fp32;
    *matrix[1][6][1] = 	(ukrFunction)gemm_RISCV_1x6_b1_col_fp32;
    *matrix[1][7][0] = 	(ukrFunction)gemm_RISCV_1x7_b0_col_fp32;
    *matrix[1][7][1] = 	(ukrFunction)gemm_RISCV_1x7_b1_col_fp32;
    *matrix[1][8][0] = 	(ukrFunction)gemm_RISCV_1x8_b0_col_fp32;
    *matrix[1][8][1] = 	(ukrFunction)gemm_RISCV_1x8_b1_col_fp32;
    *matrix[1][9][0] = 	(ukrFunction)gemm_RISCV_1x9_b0_col_fp32;
    *matrix[1][9][1] = 	(ukrFunction)gemm_RISCV_1x9_b1_col_fp32;
    *matrix[1][10][0] = 	(ukrFunction)gemm_RISCV_1x10_b0_col_fp32;
    *matrix[1][10][1] = 	(ukrFunction)gemm_RISCV_1x10_b1_col_fp32;
    *matrix[1][11][0] = 	(ukrFunction)gemm_RISCV_1x11_b0_col_fp32;
    *matrix[1][11][1] = 	(ukrFunction)gemm_RISCV_1x11_b1_col_fp32;
    *matrix[1][12][0] = 	(ukrFunction)gemm_RISCV_1x12_b0_col_fp32;
    *matrix[1][12][1] = 	(ukrFunction)gemm_RISCV_1x12_b1_col_fp32;
    *matrix[1][13][0] = 	(ukrFunction)gemm_RISCV_1x13_b0_col_fp32;
    *matrix[1][13][1] = 	(ukrFunction)gemm_RISCV_1x13_b1_col_fp32;
    *matrix[1][14][0] = 	(ukrFunction)gemm_RISCV_1x14_b0_col_fp32;
    *matrix[1][14][1] = 	(ukrFunction)gemm_RISCV_1x14_b1_col_fp32;
    *matrix[1][15][0] = 	(ukrFunction)gemm_RISCV_1x15_b0_col_fp32;
    *matrix[1][15][1] = 	(ukrFunction)gemm_RISCV_1x15_b1_col_fp32;
    *matrix[1][16][0] = 	(ukrFunction)gemm_RISCV_1x16_b0_col_fp32;
    *matrix[1][16][1] = 	(ukrFunction)gemm_RISCV_1x16_b1_col_fp32;
    *matrix[1][17][0] = 	(ukrFunction)gemm_RISCV_1x17_b0_col_fp32;
    *matrix[1][17][1] = 	(ukrFunction)gemm_RISCV_1x17_b1_col_fp32;
    *matrix[1][18][0] = 	(ukrFunction)gemm_RISCV_1x18_b0_col_fp32;
    *matrix[1][18][1] = 	(ukrFunction)gemm_RISCV_1x18_b1_col_fp32;
    *matrix[1][19][0] = 	(ukrFunction)gemm_RISCV_1x19_b0_col_fp32;
    *matrix[1][19][1] = 	(ukrFunction)gemm_RISCV_1x19_b1_col_fp32;
    *matrix[1][20][0] = 	(ukrFunction)gemm_RISCV_1x20_b0_col_fp32;
    *matrix[1][20][1] = 	(ukrFunction)gemm_RISCV_1x20_b1_col_fp32;
    *matrix[1][21][0] = 	(ukrFunction)gemm_RISCV_1x21_b0_col_fp32;
    *matrix[1][21][1] = 	(ukrFunction)gemm_RISCV_1x21_b1_col_fp32;
    *matrix[1][22][0] = 	(ukrFunction)gemm_RISCV_1x22_b0_col_fp32;
    *matrix[1][22][1] = 	(ukrFunction)gemm_RISCV_1x22_b1_col_fp32;
    *matrix[1][23][0] = 	(ukrFunction)gemm_RISCV_1x23_b0_col_fp32;
    *matrix[1][23][1] = 	(ukrFunction)gemm_RISCV_1x23_b1_col_fp32;
    *matrix[1][24][0] = 	(ukrFunction)gemm_RISCV_1x24_b0_col_fp32;
    *matrix[1][24][1] = 	(ukrFunction)gemm_RISCV_1x24_b1_col_fp32;
    *matrix[2][0][0] = 	(ukrFunction)NULL;
    *matrix[2][0][1] = 	(ukrFunction)NULL;
    *matrix[2][1][0] = 	(ukrFunction)gemm_RISCV_2x1_b0_col_fp32;
    *matrix[2][1][1] = 	(ukrFunction)gemm_RISCV_2x1_b1_col_fp32;
    *matrix[2][2][0] = 	(ukrFunction)gemm_RISCV_2x2_b0_col_fp32;
    *matrix[2][2][1] = 	(ukrFunction)gemm_RISCV_2x2_b1_col_fp32;
    *matrix[2][3][0] = 	(ukrFunction)gemm_RISCV_2x3_b0_col_fp32;
    *matrix[2][3][1] = 	(ukrFunction)gemm_RISCV_2x3_b1_col_fp32;
    *matrix[2][4][0] = 	(ukrFunction)gemm_RISCV_2x4_b0_col_fp32;
    *matrix[2][4][1] = 	(ukrFunction)gemm_RISCV_2x4_b1_col_fp32;
    *matrix[2][5][0] = 	(ukrFunction)gemm_RISCV_2x5_b0_col_fp32;
    *matrix[2][5][1] = 	(ukrFunction)gemm_RISCV_2x5_b1_col_fp32;
    *matrix[2][6][0] = 	(ukrFunction)gemm_RISCV_2x6_b0_col_fp32;
    *matrix[2][6][1] = 	(ukrFunction)gemm_RISCV_2x6_b1_col_fp32;
    *matrix[2][7][0] = 	(ukrFunction)gemm_RISCV_2x7_b0_col_fp32;
    *matrix[2][7][1] = 	(ukrFunction)gemm_RISCV_2x7_b1_col_fp32;
    *matrix[2][8][0] = 	(ukrFunction)gemm_RISCV_2x8_b0_col_fp32;
    *matrix[2][8][1] = 	(ukrFunction)gemm_RISCV_2x8_b1_col_fp32;
    *matrix[2][9][0] = 	(ukrFunction)gemm_RISCV_2x9_b0_col_fp32;
    *matrix[2][9][1] = 	(ukrFunction)gemm_RISCV_2x9_b1_col_fp32;
    *matrix[2][10][0] = 	(ukrFunction)gemm_RISCV_2x10_b0_col_fp32;
    *matrix[2][10][1] = 	(ukrFunction)gemm_RISCV_2x10_b1_col_fp32;
    *matrix[2][11][0] = 	(ukrFunction)gemm_RISCV_2x11_b0_col_fp32;
    *matrix[2][11][1] = 	(ukrFunction)gemm_RISCV_2x11_b1_col_fp32;
    *matrix[2][12][0] = 	(ukrFunction)gemm_RISCV_2x12_b0_col_fp32;
    *matrix[2][12][1] = 	(ukrFunction)gemm_RISCV_2x12_b1_col_fp32;
    *matrix[2][13][0] = 	(ukrFunction)gemm_RISCV_2x13_b0_col_fp32;
    *matrix[2][13][1] = 	(ukrFunction)gemm_RISCV_2x13_b1_col_fp32;
    *matrix[2][14][0] = 	(ukrFunction)gemm_RISCV_2x14_b0_col_fp32;
    *matrix[2][14][1] = 	(ukrFunction)gemm_RISCV_2x14_b1_col_fp32;
    *matrix[2][15][0] = 	(ukrFunction)gemm_RISCV_2x15_b0_col_fp32;
    *matrix[2][15][1] = 	(ukrFunction)gemm_RISCV_2x15_b1_col_fp32;
    *matrix[2][16][0] = 	(ukrFunction)gemm_RISCV_2x16_b0_col_fp32;
    *matrix[2][16][1] = 	(ukrFunction)gemm_RISCV_2x16_b1_col_fp32;
    *matrix[2][17][0] = 	(ukrFunction)gemm_RISCV_2x17_b0_col_fp32;
    *matrix[2][17][1] = 	(ukrFunction)gemm_RISCV_2x17_b1_col_fp32;
    *matrix[2][18][0] = 	(ukrFunction)gemm_RISCV_2x18_b0_col_fp32;
    *matrix[2][18][1] = 	(ukrFunction)gemm_RISCV_2x18_b1_col_fp32;
    *matrix[2][19][0] = 	(ukrFunction)gemm_RISCV_2x19_b0_col_fp32;
    *matrix[2][19][1] = 	(ukrFunction)gemm_RISCV_2x19_b1_col_fp32;
    *matrix[2][20][0] = 	(ukrFunction)gemm_RISCV_2x20_b0_col_fp32;
    *matrix[2][20][1] = 	(ukrFunction)gemm_RISCV_2x20_b1_col_fp32;
    *matrix[2][21][0] = 	(ukrFunction)gemm_RISCV_2x21_b0_col_fp32;
    *matrix[2][21][1] = 	(ukrFunction)gemm_RISCV_2x21_b1_col_fp32;
    *matrix[2][22][0] = 	(ukrFunction)gemm_RISCV_2x22_b0_col_fp32;
    *matrix[2][22][1] = 	(ukrFunction)gemm_RISCV_2x22_b1_col_fp32;
    *matrix[2][23][0] = 	(ukrFunction)gemm_RISCV_2x23_b0_col_fp32;
    *matrix[2][23][1] = 	(ukrFunction)gemm_RISCV_2x23_b1_col_fp32;
    *matrix[2][24][0] = 	(ukrFunction)gemm_RISCV_2x24_b0_col_fp32;
    *matrix[2][24][1] = 	(ukrFunction)gemm_RISCV_2x24_b1_col_fp32;
    *matrix[3][0][0] = 	(ukrFunction)NULL;
    *matrix[3][0][1] = 	(ukrFunction)NULL;
    *matrix[3][1][0] = 	(ukrFunction)gemm_RISCV_3x1_b0_col_fp32;
    *matrix[3][1][1] = 	(ukrFunction)gemm_RISCV_3x1_b1_col_fp32;
    *matrix[3][2][0] = 	(ukrFunction)gemm_RISCV_3x2_b0_col_fp32;
    *matrix[3][2][1] = 	(ukrFunction)gemm_RISCV_3x2_b1_col_fp32;
    *matrix[3][3][0] = 	(ukrFunction)gemm_RISCV_3x3_b0_col_fp32;
    *matrix[3][3][1] = 	(ukrFunction)gemm_RISCV_3x3_b1_col_fp32;
    *matrix[3][4][0] = 	(ukrFunction)gemm_RISCV_3x4_b0_col_fp32;
    *matrix[3][4][1] = 	(ukrFunction)gemm_RISCV_3x4_b1_col_fp32;
    *matrix[3][5][0] = 	(ukrFunction)gemm_RISCV_3x5_b0_col_fp32;
    *matrix[3][5][1] = 	(ukrFunction)gemm_RISCV_3x5_b1_col_fp32;
    *matrix[3][6][0] = 	(ukrFunction)gemm_RISCV_3x6_b0_col_fp32;
    *matrix[3][6][1] = 	(ukrFunction)gemm_RISCV_3x6_b1_col_fp32;
    *matrix[3][7][0] = 	(ukrFunction)gemm_RISCV_3x7_b0_col_fp32;
    *matrix[3][7][1] = 	(ukrFunction)gemm_RISCV_3x7_b1_col_fp32;
    *matrix[3][8][0] = 	(ukrFunction)gemm_RISCV_3x8_b0_col_fp32;
    *matrix[3][8][1] = 	(ukrFunction)gemm_RISCV_3x8_b1_col_fp32;
    *matrix[3][9][0] = 	(ukrFunction)gemm_RISCV_3x9_b0_col_fp32;
    *matrix[3][9][1] = 	(ukrFunction)gemm_RISCV_3x9_b1_col_fp32;
    *matrix[3][10][0] = 	(ukrFunction)gemm_RISCV_3x10_b0_col_fp32;
    *matrix[3][10][1] = 	(ukrFunction)gemm_RISCV_3x10_b1_col_fp32;
    *matrix[3][11][0] = 	(ukrFunction)gemm_RISCV_3x11_b0_col_fp32;
    *matrix[3][11][1] = 	(ukrFunction)gemm_RISCV_3x11_b1_col_fp32;
    *matrix[3][12][0] = 	(ukrFunction)gemm_RISCV_3x12_b0_col_fp32;
    *matrix[3][12][1] = 	(ukrFunction)gemm_RISCV_3x12_b1_col_fp32;
    *matrix[3][13][0] = 	(ukrFunction)gemm_RISCV_3x13_b0_col_fp32;
    *matrix[3][13][1] = 	(ukrFunction)gemm_RISCV_3x13_b1_col_fp32;
    *matrix[3][14][0] = 	(ukrFunction)gemm_RISCV_3x14_b0_col_fp32;
    *matrix[3][14][1] = 	(ukrFunction)gemm_RISCV_3x14_b1_col_fp32;
    *matrix[3][15][0] = 	(ukrFunction)gemm_RISCV_3x15_b0_col_fp32;
    *matrix[3][15][1] = 	(ukrFunction)gemm_RISCV_3x15_b1_col_fp32;
    *matrix[3][16][0] = 	(ukrFunction)gemm_RISCV_3x16_b0_col_fp32;
    *matrix[3][16][1] = 	(ukrFunction)gemm_RISCV_3x16_b1_col_fp32;
    *matrix[3][17][0] = 	(ukrFunction)gemm_RISCV_3x17_b0_col_fp32;
    *matrix[3][17][1] = 	(ukrFunction)gemm_RISCV_3x17_b1_col_fp32;
    *matrix[3][18][0] = 	(ukrFunction)gemm_RISCV_3x18_b0_col_fp32;
    *matrix[3][18][1] = 	(ukrFunction)gemm_RISCV_3x18_b1_col_fp32;
    *matrix[3][19][0] = 	(ukrFunction)gemm_RISCV_3x19_b0_col_fp32;
    *matrix[3][19][1] = 	(ukrFunction)gemm_RISCV_3x19_b1_col_fp32;
    *matrix[3][20][0] = 	(ukrFunction)gemm_RISCV_3x20_b0_col_fp32;
    *matrix[3][20][1] = 	(ukrFunction)gemm_RISCV_3x20_b1_col_fp32;
    *matrix[3][21][0] = 	(ukrFunction)gemm_RISCV_3x21_b0_col_fp32;
    *matrix[3][21][1] = 	(ukrFunction)gemm_RISCV_3x21_b1_col_fp32;
    *matrix[3][22][0] = 	(ukrFunction)gemm_RISCV_3x22_b0_col_fp32;
    *matrix[3][22][1] = 	(ukrFunction)gemm_RISCV_3x22_b1_col_fp32;
    *matrix[3][23][0] = 	(ukrFunction)gemm_RISCV_3x23_b0_col_fp32;
    *matrix[3][23][1] = 	(ukrFunction)gemm_RISCV_3x23_b1_col_fp32;
    *matrix[3][24][0] = 	(ukrFunction)gemm_RISCV_3x24_b0_col_fp32;
    *matrix[3][24][1] = 	(ukrFunction)gemm_RISCV_3x24_b1_col_fp32;
    *matrix[4][0][0] = 	(ukrFunction)NULL;
    *matrix[4][0][1] = 	(ukrFunction)NULL;
    *matrix[4][1][0] = 	(ukrFunction)gemm_RISCV_4x1_b0_col_fp32;
    *matrix[4][1][1] = 	(ukrFunction)gemm_RISCV_4x1_b1_col_fp32;
    *matrix[4][2][0] = 	(ukrFunction)gemm_RISCV_4x2_b0_col_fp32;
    *matrix[4][2][1] = 	(ukrFunction)gemm_RISCV_4x2_b1_col_fp32;
    *matrix[4][3][0] = 	(ukrFunction)gemm_RISCV_4x3_b0_col_fp32;
    *matrix[4][3][1] = 	(ukrFunction)gemm_RISCV_4x3_b1_col_fp32;
    *matrix[4][4][0] = 	(ukrFunction)gemm_RISCV_4x4_b0_col_fp32;
    *matrix[4][4][1] = 	(ukrFunction)gemm_RISCV_4x4_b1_col_fp32;
    *matrix[4][5][0] = 	(ukrFunction)gemm_RISCV_4x5_b0_col_fp32;
    *matrix[4][5][1] = 	(ukrFunction)gemm_RISCV_4x5_b1_col_fp32;
    *matrix[4][6][0] = 	(ukrFunction)gemm_RISCV_4x6_b0_col_fp32;
    *matrix[4][6][1] = 	(ukrFunction)gemm_RISCV_4x6_b1_col_fp32;
    *matrix[4][7][0] = 	(ukrFunction)gemm_RISCV_4x7_b0_col_fp32;
    *matrix[4][7][1] = 	(ukrFunction)gemm_RISCV_4x7_b1_col_fp32;
    *matrix[4][8][0] = 	(ukrFunction)gemm_RISCV_4x8_b0_col_fp32;
    *matrix[4][8][1] = 	(ukrFunction)gemm_RISCV_4x8_b1_col_fp32;
    *matrix[4][9][0] = 	(ukrFunction)gemm_RISCV_4x9_b0_col_fp32;
    *matrix[4][9][1] = 	(ukrFunction)gemm_RISCV_4x9_b1_col_fp32;
    *matrix[4][10][0] = 	(ukrFunction)gemm_RISCV_4x10_b0_col_fp32;
    *matrix[4][10][1] = 	(ukrFunction)gemm_RISCV_4x10_b1_col_fp32;
    *matrix[4][11][0] = 	(ukrFunction)gemm_RISCV_4x11_b0_col_fp32;
    *matrix[4][11][1] = 	(ukrFunction)gemm_RISCV_4x11_b1_col_fp32;
    *matrix[4][12][0] = 	(ukrFunction)gemm_RISCV_4x12_b0_col_fp32;
    *matrix[4][12][1] = 	(ukrFunction)gemm_RISCV_4x12_b1_col_fp32;
    *matrix[4][13][0] = 	(ukrFunction)gemm_RISCV_4x13_b0_col_fp32;
    *matrix[4][13][1] = 	(ukrFunction)gemm_RISCV_4x13_b1_col_fp32;
    *matrix[4][14][0] = 	(ukrFunction)gemm_RISCV_4x14_b0_col_fp32;
    *matrix[4][14][1] = 	(ukrFunction)gemm_RISCV_4x14_b1_col_fp32;
    *matrix[4][15][0] = 	(ukrFunction)gemm_RISCV_4x15_b0_col_fp32;
    *matrix[4][15][1] = 	(ukrFunction)gemm_RISCV_4x15_b1_col_fp32;
    *matrix[4][16][0] = 	(ukrFunction)gemm_RISCV_4x16_b0_col_fp32;
    *matrix[4][16][1] = 	(ukrFunction)gemm_RISCV_4x16_b1_col_fp32;
    *matrix[4][17][0] = 	(ukrFunction)gemm_RISCV_4x17_b0_col_fp32;
    *matrix[4][17][1] = 	(ukrFunction)gemm_RISCV_4x17_b1_col_fp32;
    *matrix[4][18][0] = 	(ukrFunction)gemm_RISCV_4x18_b0_col_fp32;
    *matrix[4][18][1] = 	(ukrFunction)gemm_RISCV_4x18_b1_col_fp32;
    *matrix[4][19][0] = 	(ukrFunction)gemm_RISCV_4x19_b0_col_fp32;
    *matrix[4][19][1] = 	(ukrFunction)gemm_RISCV_4x19_b1_col_fp32;
    *matrix[4][20][0] = 	(ukrFunction)gemm_RISCV_4x20_b0_col_fp32;
    *matrix[4][20][1] = 	(ukrFunction)gemm_RISCV_4x20_b1_col_fp32;
    *matrix[4][21][0] = 	(ukrFunction)gemm_RISCV_4x21_b0_col_fp32;
    *matrix[4][21][1] = 	(ukrFunction)gemm_RISCV_4x21_b1_col_fp32;
    *matrix[4][22][0] = 	(ukrFunction)gemm_RISCV_4x22_b0_col_fp32;
    *matrix[4][22][1] = 	(ukrFunction)gemm_RISCV_4x22_b1_col_fp32;
    *matrix[4][23][0] = 	(ukrFunction)gemm_RISCV_4x23_b0_col_fp32;
    *matrix[4][23][1] = 	(ukrFunction)gemm_RISCV_4x23_b1_col_fp32;
    *matrix[4][24][0] = 	(ukrFunction)gemm_RISCV_4x24_b0_col_fp32;
    *matrix[4][24][1] = 	(ukrFunction)gemm_RISCV_4x24_b1_col_fp32;
    *matrix[5][0][0] = 	(ukrFunction)NULL;
    *matrix[5][0][1] = 	(ukrFunction)NULL;
    *matrix[5][1][0] = 	(ukrFunction)gemm_RISCV_5x1_b0_col_fp32;
    *matrix[5][1][1] = 	(ukrFunction)gemm_RISCV_5x1_b1_col_fp32;
    *matrix[5][2][0] = 	(ukrFunction)gemm_RISCV_5x2_b0_col_fp32;
    *matrix[5][2][1] = 	(ukrFunction)gemm_RISCV_5x2_b1_col_fp32;
    *matrix[5][3][0] = 	(ukrFunction)gemm_RISCV_5x3_b0_col_fp32;
    *matrix[5][3][1] = 	(ukrFunction)gemm_RISCV_5x3_b1_col_fp32;
    *matrix[5][4][0] = 	(ukrFunction)gemm_RISCV_5x4_b0_col_fp32;
    *matrix[5][4][1] = 	(ukrFunction)gemm_RISCV_5x4_b1_col_fp32;
    *matrix[5][5][0] = 	(ukrFunction)gemm_RISCV_5x5_b0_col_fp32;
    *matrix[5][5][1] = 	(ukrFunction)gemm_RISCV_5x5_b1_col_fp32;
    *matrix[5][6][0] = 	(ukrFunction)gemm_RISCV_5x6_b0_col_fp32;
    *matrix[5][6][1] = 	(ukrFunction)gemm_RISCV_5x6_b1_col_fp32;
    *matrix[5][7][0] = 	(ukrFunction)gemm_RISCV_5x7_b0_col_fp32;
    *matrix[5][7][1] = 	(ukrFunction)gemm_RISCV_5x7_b1_col_fp32;
    *matrix[5][8][0] = 	(ukrFunction)gemm_RISCV_5x8_b0_col_fp32;
    *matrix[5][8][1] = 	(ukrFunction)gemm_RISCV_5x8_b1_col_fp32;
    *matrix[5][9][0] = 	(ukrFunction)gemm_RISCV_5x9_b0_col_fp32;
    *matrix[5][9][1] = 	(ukrFunction)gemm_RISCV_5x9_b1_col_fp32;
    *matrix[5][10][0] = 	(ukrFunction)gemm_RISCV_5x10_b0_col_fp32;
    *matrix[5][10][1] = 	(ukrFunction)gemm_RISCV_5x10_b1_col_fp32;
    *matrix[5][11][0] = 	(ukrFunction)gemm_RISCV_5x11_b0_col_fp32;
    *matrix[5][11][1] = 	(ukrFunction)gemm_RISCV_5x11_b1_col_fp32;
    *matrix[5][12][0] = 	(ukrFunction)gemm_RISCV_5x12_b0_col_fp32;
    *matrix[5][12][1] = 	(ukrFunction)gemm_RISCV_5x12_b1_col_fp32;
    *matrix[5][13][0] = 	(ukrFunction)gemm_RISCV_5x13_b0_col_fp32;
    *matrix[5][13][1] = 	(ukrFunction)gemm_RISCV_5x13_b1_col_fp32;
    *matrix[5][14][0] = 	(ukrFunction)gemm_RISCV_5x14_b0_col_fp32;
    *matrix[5][14][1] = 	(ukrFunction)gemm_RISCV_5x14_b1_col_fp32;
    *matrix[5][15][0] = 	(ukrFunction)gemm_RISCV_5x15_b0_col_fp32;
    *matrix[5][15][1] = 	(ukrFunction)gemm_RISCV_5x15_b1_col_fp32;
    *matrix[5][16][0] = 	(ukrFunction)gemm_RISCV_5x16_b0_col_fp32;
    *matrix[5][16][1] = 	(ukrFunction)gemm_RISCV_5x16_b1_col_fp32;
    *matrix[5][17][0] = 	(ukrFunction)gemm_RISCV_5x17_b0_col_fp32;
    *matrix[5][17][1] = 	(ukrFunction)gemm_RISCV_5x17_b1_col_fp32;
    *matrix[5][18][0] = 	(ukrFunction)gemm_RISCV_5x18_b0_col_fp32;
    *matrix[5][18][1] = 	(ukrFunction)gemm_RISCV_5x18_b1_col_fp32;
    *matrix[5][19][0] = 	(ukrFunction)gemm_RISCV_5x19_b0_col_fp32;
    *matrix[5][19][1] = 	(ukrFunction)gemm_RISCV_5x19_b1_col_fp32;
    *matrix[5][20][0] = 	(ukrFunction)gemm_RISCV_5x20_b0_col_fp32;
    *matrix[5][20][1] = 	(ukrFunction)gemm_RISCV_5x20_b1_col_fp32;
    *matrix[5][21][0] = 	(ukrFunction)gemm_RISCV_5x21_b0_col_fp32;
    *matrix[5][21][1] = 	(ukrFunction)gemm_RISCV_5x21_b1_col_fp32;
    *matrix[5][22][0] = 	(ukrFunction)gemm_RISCV_5x22_b0_col_fp32;
    *matrix[5][22][1] = 	(ukrFunction)gemm_RISCV_5x22_b1_col_fp32;
    *matrix[5][23][0] = 	(ukrFunction)gemm_RISCV_5x23_b0_col_fp32;
    *matrix[5][23][1] = 	(ukrFunction)gemm_RISCV_5x23_b1_col_fp32;
    *matrix[5][24][0] = 	(ukrFunction)gemm_RISCV_5x24_b0_col_fp32;
    *matrix[5][24][1] = 	(ukrFunction)gemm_RISCV_5x24_b1_col_fp32;
    *matrix[6][0][0] = 	(ukrFunction)NULL;
    *matrix[6][0][1] = 	(ukrFunction)NULL;
    *matrix[6][1][0] = 	(ukrFunction)gemm_RISCV_6x1_b0_col_fp32;
    *matrix[6][1][1] = 	(ukrFunction)gemm_RISCV_6x1_b1_col_fp32;
    *matrix[6][2][0] = 	(ukrFunction)gemm_RISCV_6x2_b0_col_fp32;
    *matrix[6][2][1] = 	(ukrFunction)gemm_RISCV_6x2_b1_col_fp32;
    *matrix[6][3][0] = 	(ukrFunction)gemm_RISCV_6x3_b0_col_fp32;
    *matrix[6][3][1] = 	(ukrFunction)gemm_RISCV_6x3_b1_col_fp32;
    *matrix[6][4][0] = 	(ukrFunction)gemm_RISCV_6x4_b0_col_fp32;
    *matrix[6][4][1] = 	(ukrFunction)gemm_RISCV_6x4_b1_col_fp32;
    *matrix[6][5][0] = 	(ukrFunction)gemm_RISCV_6x5_b0_col_fp32;
    *matrix[6][5][1] = 	(ukrFunction)gemm_RISCV_6x5_b1_col_fp32;
    *matrix[6][6][0] = 	(ukrFunction)gemm_RISCV_6x6_b0_col_fp32;
    *matrix[6][6][1] = 	(ukrFunction)gemm_RISCV_6x6_b1_col_fp32;
    *matrix[6][7][0] = 	(ukrFunction)gemm_RISCV_6x7_b0_col_fp32;
    *matrix[6][7][1] = 	(ukrFunction)gemm_RISCV_6x7_b1_col_fp32;
    *matrix[6][8][0] = 	(ukrFunction)gemm_RISCV_6x8_b0_col_fp32;
    *matrix[6][8][1] = 	(ukrFunction)gemm_RISCV_6x8_b1_col_fp32;
    *matrix[6][9][0] = 	(ukrFunction)gemm_RISCV_6x9_b0_col_fp32;
    *matrix[6][9][1] = 	(ukrFunction)gemm_RISCV_6x9_b1_col_fp32;
    *matrix[6][10][0] = 	(ukrFunction)gemm_RISCV_6x10_b0_col_fp32;
    *matrix[6][10][1] = 	(ukrFunction)gemm_RISCV_6x10_b1_col_fp32;
    *matrix[6][11][0] = 	(ukrFunction)gemm_RISCV_6x11_b0_col_fp32;
    *matrix[6][11][1] = 	(ukrFunction)gemm_RISCV_6x11_b1_col_fp32;
    *matrix[6][12][0] = 	(ukrFunction)gemm_RISCV_6x12_b0_col_fp32;
    *matrix[6][12][1] = 	(ukrFunction)gemm_RISCV_6x12_b1_col_fp32;
    *matrix[6][13][0] = 	(ukrFunction)gemm_RISCV_6x13_b0_col_fp32;
    *matrix[6][13][1] = 	(ukrFunction)gemm_RISCV_6x13_b1_col_fp32;
    *matrix[6][14][0] = 	(ukrFunction)gemm_RISCV_6x14_b0_col_fp32;
    *matrix[6][14][1] = 	(ukrFunction)gemm_RISCV_6x14_b1_col_fp32;
    *matrix[6][15][0] = 	(ukrFunction)gemm_RISCV_6x15_b0_col_fp32;
    *matrix[6][15][1] = 	(ukrFunction)gemm_RISCV_6x15_b1_col_fp32;
    *matrix[6][16][0] = 	(ukrFunction)gemm_RISCV_6x16_b0_col_fp32;
    *matrix[6][16][1] = 	(ukrFunction)gemm_RISCV_6x16_b1_col_fp32;
    *matrix[6][17][0] = 	(ukrFunction)gemm_RISCV_6x17_b0_col_fp32;
    *matrix[6][17][1] = 	(ukrFunction)gemm_RISCV_6x17_b1_col_fp32;
    *matrix[6][18][0] = 	(ukrFunction)gemm_RISCV_6x18_b0_col_fp32;
    *matrix[6][18][1] = 	(ukrFunction)gemm_RISCV_6x18_b1_col_fp32;
    *matrix[6][19][0] = 	(ukrFunction)gemm_RISCV_6x19_b0_col_fp32;
    *matrix[6][19][1] = 	(ukrFunction)gemm_RISCV_6x19_b1_col_fp32;
    *matrix[6][20][0] = 	(ukrFunction)gemm_RISCV_6x20_b0_col_fp32;
    *matrix[6][20][1] = 	(ukrFunction)gemm_RISCV_6x20_b1_col_fp32;
    *matrix[6][21][0] = 	(ukrFunction)gemm_RISCV_6x21_b0_col_fp32;
    *matrix[6][21][1] = 	(ukrFunction)gemm_RISCV_6x21_b1_col_fp32;
    *matrix[6][22][0] = 	(ukrFunction)gemm_RISCV_6x22_b0_col_fp32;
    *matrix[6][22][1] = 	(ukrFunction)gemm_RISCV_6x22_b1_col_fp32;
    *matrix[6][23][0] = 	(ukrFunction)gemm_RISCV_6x23_b0_col_fp32;
    *matrix[6][23][1] = 	(ukrFunction)gemm_RISCV_6x23_b1_col_fp32;
    *matrix[6][24][0] = 	(ukrFunction)gemm_RISCV_6x24_b0_col_fp32;
    *matrix[6][24][1] = 	(ukrFunction)gemm_RISCV_6x24_b1_col_fp32;
    *matrix[7][0][0] = 	(ukrFunction)NULL;
    *matrix[7][0][1] = 	(ukrFunction)NULL;
    *matrix[7][1][0] = 	(ukrFunction)gemm_RISCV_7x1_b0_col_fp32;
    *matrix[7][1][1] = 	(ukrFunction)gemm_RISCV_7x1_b1_col_fp32;
    *matrix[7][2][0] = 	(ukrFunction)gemm_RISCV_7x2_b0_col_fp32;
    *matrix[7][2][1] = 	(ukrFunction)gemm_RISCV_7x2_b1_col_fp32;
    *matrix[7][3][0] = 	(ukrFunction)gemm_RISCV_7x3_b0_col_fp32;
    *matrix[7][3][1] = 	(ukrFunction)gemm_RISCV_7x3_b1_col_fp32;
    *matrix[7][4][0] = 	(ukrFunction)gemm_RISCV_7x4_b0_col_fp32;
    *matrix[7][4][1] = 	(ukrFunction)gemm_RISCV_7x4_b1_col_fp32;
    *matrix[7][5][0] = 	(ukrFunction)gemm_RISCV_7x5_b0_col_fp32;
    *matrix[7][5][1] = 	(ukrFunction)gemm_RISCV_7x5_b1_col_fp32;
    *matrix[7][6][0] = 	(ukrFunction)gemm_RISCV_7x6_b0_col_fp32;
    *matrix[7][6][1] = 	(ukrFunction)gemm_RISCV_7x6_b1_col_fp32;
    *matrix[7][7][0] = 	(ukrFunction)gemm_RISCV_7x7_b0_col_fp32;
    *matrix[7][7][1] = 	(ukrFunction)gemm_RISCV_7x7_b1_col_fp32;
    *matrix[7][8][0] = 	(ukrFunction)gemm_RISCV_7x8_b0_col_fp32;
    *matrix[7][8][1] = 	(ukrFunction)gemm_RISCV_7x8_b1_col_fp32;
    *matrix[7][9][0] = 	(ukrFunction)gemm_RISCV_7x9_b0_col_fp32;
    *matrix[7][9][1] = 	(ukrFunction)gemm_RISCV_7x9_b1_col_fp32;
    *matrix[7][10][0] = 	(ukrFunction)gemm_RISCV_7x10_b0_col_fp32;
    *matrix[7][10][1] = 	(ukrFunction)gemm_RISCV_7x10_b1_col_fp32;
    *matrix[7][11][0] = 	(ukrFunction)gemm_RISCV_7x11_b0_col_fp32;
    *matrix[7][11][1] = 	(ukrFunction)gemm_RISCV_7x11_b1_col_fp32;
    *matrix[7][12][0] = 	(ukrFunction)gemm_RISCV_7x12_b0_col_fp32;
    *matrix[7][12][1] = 	(ukrFunction)gemm_RISCV_7x12_b1_col_fp32;
    *matrix[7][13][0] = 	(ukrFunction)gemm_RISCV_7x13_b0_col_fp32;
    *matrix[7][13][1] = 	(ukrFunction)gemm_RISCV_7x13_b1_col_fp32;
    *matrix[7][14][0] = 	(ukrFunction)gemm_RISCV_7x14_b0_col_fp32;
    *matrix[7][14][1] = 	(ukrFunction)gemm_RISCV_7x14_b1_col_fp32;
    *matrix[7][15][0] = 	(ukrFunction)gemm_RISCV_7x15_b0_col_fp32;
    *matrix[7][15][1] = 	(ukrFunction)gemm_RISCV_7x15_b1_col_fp32;
    *matrix[7][16][0] = 	(ukrFunction)gemm_RISCV_7x16_b0_col_fp32;
    *matrix[7][16][1] = 	(ukrFunction)gemm_RISCV_7x16_b1_col_fp32;
    *matrix[7][17][0] = 	(ukrFunction)gemm_RISCV_7x17_b0_col_fp32;
    *matrix[7][17][1] = 	(ukrFunction)gemm_RISCV_7x17_b1_col_fp32;
    *matrix[7][18][0] = 	(ukrFunction)gemm_RISCV_7x18_b0_col_fp32;
    *matrix[7][18][1] = 	(ukrFunction)gemm_RISCV_7x18_b1_col_fp32;
    *matrix[7][19][0] = 	(ukrFunction)gemm_RISCV_7x19_b0_col_fp32;
    *matrix[7][19][1] = 	(ukrFunction)gemm_RISCV_7x19_b1_col_fp32;
    *matrix[7][20][0] = 	(ukrFunction)gemm_RISCV_7x20_b0_col_fp32;
    *matrix[7][20][1] = 	(ukrFunction)gemm_RISCV_7x20_b1_col_fp32;
    *matrix[7][21][0] = 	(ukrFunction)gemm_RISCV_7x21_b0_col_fp32;
    *matrix[7][21][1] = 	(ukrFunction)gemm_RISCV_7x21_b1_col_fp32;
    *matrix[7][22][0] = 	(ukrFunction)gemm_RISCV_7x22_b0_col_fp32;
    *matrix[7][22][1] = 	(ukrFunction)gemm_RISCV_7x22_b1_col_fp32;
    *matrix[7][23][0] = 	(ukrFunction)gemm_RISCV_7x23_b0_col_fp32;
    *matrix[7][23][1] = 	(ukrFunction)gemm_RISCV_7x23_b1_col_fp32;
    *matrix[7][24][0] = 	(ukrFunction)gemm_RISCV_7x24_b0_col_fp32;
    *matrix[7][24][1] = 	(ukrFunction)gemm_RISCV_7x24_b1_col_fp32;
    *matrix[8][0][0] = 	(ukrFunction)NULL;
    *matrix[8][0][1] = 	(ukrFunction)NULL;
    *matrix[8][1][0] = 	(ukrFunction)gemm_RISCV_8x1_b0_col_fp32;
    *matrix[8][1][1] = 	(ukrFunction)gemm_RISCV_8x1_b1_col_fp32;
    *matrix[8][2][0] = 	(ukrFunction)gemm_RISCV_8x2_b0_col_fp32;
    *matrix[8][2][1] = 	(ukrFunction)gemm_RISCV_8x2_b1_col_fp32;
    *matrix[8][3][0] = 	(ukrFunction)gemm_RISCV_8x3_b0_col_fp32;
    *matrix[8][3][1] = 	(ukrFunction)gemm_RISCV_8x3_b1_col_fp32;
    *matrix[8][4][0] = 	(ukrFunction)gemm_RISCV_8x4_b0_col_fp32;
    *matrix[8][4][1] = 	(ukrFunction)gemm_RISCV_8x4_b1_col_fp32;
    *matrix[8][5][0] = 	(ukrFunction)gemm_RISCV_8x5_b0_col_fp32;
    *matrix[8][5][1] = 	(ukrFunction)gemm_RISCV_8x5_b1_col_fp32;
    *matrix[8][6][0] = 	(ukrFunction)gemm_RISCV_8x6_b0_col_fp32;
    *matrix[8][6][1] = 	(ukrFunction)gemm_RISCV_8x6_b1_col_fp32;
    *matrix[8][7][0] = 	(ukrFunction)gemm_RISCV_8x7_b0_col_fp32;
    *matrix[8][7][1] = 	(ukrFunction)gemm_RISCV_8x7_b1_col_fp32;
    *matrix[8][8][0] = 	(ukrFunction)gemm_RISCV_8x8_b0_col_fp32;
    *matrix[8][8][1] = 	(ukrFunction)gemm_RISCV_8x8_b1_col_fp32;
    *matrix[8][9][0] = 	(ukrFunction)gemm_RISCV_8x9_b0_col_fp32;
    *matrix[8][9][1] = 	(ukrFunction)gemm_RISCV_8x9_b1_col_fp32;
    *matrix[8][10][0] = 	(ukrFunction)gemm_RISCV_8x10_b0_col_fp32;
    *matrix[8][10][1] = 	(ukrFunction)gemm_RISCV_8x10_b1_col_fp32;
    *matrix[8][11][0] = 	(ukrFunction)gemm_RISCV_8x11_b0_col_fp32;
    *matrix[8][11][1] = 	(ukrFunction)gemm_RISCV_8x11_b1_col_fp32;
    *matrix[8][12][0] = 	(ukrFunction)gemm_RISCV_8x12_b0_col_fp32;
    *matrix[8][12][1] = 	(ukrFunction)gemm_RISCV_8x12_b1_col_fp32;
    *matrix[8][13][0] = 	(ukrFunction)gemm_RISCV_8x13_b0_col_fp32;
    *matrix[8][13][1] = 	(ukrFunction)gemm_RISCV_8x13_b1_col_fp32;
    *matrix[8][14][0] = 	(ukrFunction)gemm_RISCV_8x14_b0_col_fp32;
    *matrix[8][14][1] = 	(ukrFunction)gemm_RISCV_8x14_b1_col_fp32;
    *matrix[8][15][0] = 	(ukrFunction)gemm_RISCV_8x15_b0_col_fp32;
    *matrix[8][15][1] = 	(ukrFunction)gemm_RISCV_8x15_b1_col_fp32;
    *matrix[8][16][0] = 	(ukrFunction)gemm_RISCV_8x16_b0_col_fp32;
    *matrix[8][16][1] = 	(ukrFunction)gemm_RISCV_8x16_b1_col_fp32;
    *matrix[8][17][0] = 	(ukrFunction)gemm_RISCV_8x17_b0_col_fp32;
    *matrix[8][17][1] = 	(ukrFunction)gemm_RISCV_8x17_b1_col_fp32;
    *matrix[8][18][0] = 	(ukrFunction)gemm_RISCV_8x18_b0_col_fp32;
    *matrix[8][18][1] = 	(ukrFunction)gemm_RISCV_8x18_b1_col_fp32;
    *matrix[8][19][0] = 	(ukrFunction)gemm_RISCV_8x19_b0_col_fp32;
    *matrix[8][19][1] = 	(ukrFunction)gemm_RISCV_8x19_b1_col_fp32;
    *matrix[8][20][0] = 	(ukrFunction)gemm_RISCV_8x20_b0_col_fp32;
    *matrix[8][20][1] = 	(ukrFunction)gemm_RISCV_8x20_b1_col_fp32;
    *matrix[8][21][0] = 	(ukrFunction)gemm_RISCV_8x21_b0_col_fp32;
    *matrix[8][21][1] = 	(ukrFunction)gemm_RISCV_8x21_b1_col_fp32;
    *matrix[8][22][0] = 	(ukrFunction)gemm_RISCV_8x22_b0_col_fp32;
    *matrix[8][22][1] = 	(ukrFunction)gemm_RISCV_8x22_b1_col_fp32;
    *matrix[8][23][0] = 	(ukrFunction)gemm_RISCV_8x23_b0_col_fp32;
    *matrix[8][23][1] = 	(ukrFunction)gemm_RISCV_8x23_b1_col_fp32;
    *matrix[8][24][0] = 	(ukrFunction)gemm_RISCV_8x24_b0_col_fp32;
    *matrix[8][24][1] = 	(ukrFunction)gemm_RISCV_8x24_b1_col_fp32;
}


void freeMatrix(ukrFunction**** matrix) {
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 25; j++) {
            for (int b = 0; b < 2; b++) {
                free(matrix[i][j][b]);
            }
            free(matrix[i][j]);
        }
        free(matrix[i]);
    }
    free(matrix);
}

