#include <stdio.h>
#include <time.h>

//#include "kernel_col.h"
#include "exo_matrix.h"

#define Aref(a1,a2)  A[ (a2)*(Alda)+(a1) ]
#define Bref(a1,a2)  B[ (a2)*(Blda)+(a1) ]
#define Cref(a1,a2)  C[ (a2)*(Clda)+(a1) ]


void simplegemm(int M, int N, int K, const float * A, const float * B, float *C);
void initialize(int M, int N, int K, float * A, float *B, float *C, float *Ce);

int main(int argc, char * argv []) {
  clock_t start, end;
  float msec;
  int reps=100000;
  int M = atoi(argv[1]);
  int N = atoi(argv[2]);
  int K = atoi(argv[3]);
  float * A = malloc(sizeof(float)*M*K);
  float * B = malloc(sizeof(float)*N*K);
  float * C = malloc(sizeof(float)*M*N);
  float * Ce = malloc(sizeof(float)*M*N);
  initialize(M,N,K, A, B, C, Ce);
  ukrFunction*** ukrmatrix = allocateMatrix();
  fillMatrix(ukrmatrix);
  double gflops = (2.0*M*N*K)/1e9;
  float alpha = 1.0;
  float beta = 1.0;

  printf("TEST STARTING...!\n");
  // Calling scheduled matmul
  ukrFunction ukr = *ukrmatrix[M][N];
  start = clock();
  for (int i = 0; i < reps; i++){
      ukr(NULL, K, &alpha, A,B, &beta, (struct exo_win_2f32){Ce,{M,1}});
  }
    end = clock();

  msec = ((double)(end - start) / (double) CLOCKS_PER_SEC)/reps;
  for (int i = 0; i < reps; i++)
  simplegemm(M,N,K,A,B,C);
  for(int i = 0; i< M; i++)
  for(int j = 0; j< N; j++){
	  if(C[j* M + i] == Ce[j*M+i])
	  	 //printf("OK %f %f\n",C2[j*M+i],C3[j*M+i]);
		 continue;
	  else
	  	 printf("ERROR %f %f\n",C[j*M+i],Ce[j*M+i]);
  }
  printf("MR NR KC Time GFLOPS\n");
  printf("%d %d %d %f %f\n", M, N, K, msec, gflops/(msec*1e9));
  //printf("PASS!\n");
  return (0);
}

void simplegemm(int M, int N, int K, const float * A, const float * B, float *C){
   int Alda = M, Clda =  M;
   int Blda = N;   
   int    i, j, p;
   for ( p=0; p<K; p++ )
	   for ( j=0; j<N; j++ )
		   for ( i=0; i<M; i++ )
			   Cref(i,j) = Cref(i,j) + Aref(i,p) * Bref(j,p);
}

void initialize(int M, int N,int K,float * A, float *B, float *C, float *Ce) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < K; j++) {
      A[i * K + j] = (i * K + j);//*0.1;//3.2;
    }
  }
  for (int i = 0; i < K; i++) {
    for (int j = 0; j < N; j++) {
      B[i * N + j] = (i * N + j);//*0.2;
    }
  }
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      C[i * N + j] = 0.0;
      Ce[i * N + j] = 0.0;
    }
  }
  return;
}
