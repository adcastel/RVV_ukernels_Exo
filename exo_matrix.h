#include "kernel_col.h"
#include <stdlib.h>

typedef void (*ukr)(void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C);
typedef void (*ukrFunction)(void *ctxt, int_fast32_t KC, const float* alpha, 
		/*const float**/ struct exo_win_2f32c  A, 
		/*const float**/ struct exo_win_2f32c B, 
		const float* beta, struct exo_win_2f32 C);


ukrFunction*** allocateMatrix(); 
void fillMatrix(ukrFunction*** matrix, int beta0);
void freeMatrix(ukrFunction*** matrix); 


