#ifdef SAWPAB
    #ifdef GATHER
        #include "kernels_rvv_gather_swapAB_fp32.h"
    #else
        #include "kernels_rvv_bcast_swapAB_fp32.h"
    #endif
#else
    #ifdef GATHER
        #include "kernels_rvv_gather_fp32.h"
    #else
        #include "kernels_rvv_bcast_fp32.h"
    #endif
#endif
#include <stdlib.h>

//typedef void (*ukrFunction)(void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C);
typedef void (*ukrFunction)( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32c B, const float* beta, struct exo_win_2f32 C);

ukrFunction**** allocateMatrix(); 
void fillMatrix(ukrFunction**** matrix);
void freeMatrix(ukrFunction**** matrix); 


