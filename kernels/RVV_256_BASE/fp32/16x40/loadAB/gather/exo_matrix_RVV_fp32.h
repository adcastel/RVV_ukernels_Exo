#include "kernels_RVV_16x40_fp32.h"
#include <stdlib.h>
typedef void (*ukrFunction)( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, , struct exo_win_2f16c B, const float* beta,  struct exo_win_2f16 *C);
ukrFunction**** allocateMatrix();
void fillMatrix(ukrFunction**** matrix);
void freeMatrix(ukrFunction**** matrix);