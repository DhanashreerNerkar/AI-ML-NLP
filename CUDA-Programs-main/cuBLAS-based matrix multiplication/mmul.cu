#include <cuda.h>
#include <stdlib.h>
#include "mmul.h"
#include <cublas_v2.h>

void mmul(cublasHandle_t handle, const float* A, const float* B, float* C, int n){
    int lda=n,ldb=n,ldc=n;
    const float alf = 1;
    const float bet = 1;
    const float *alpha = &alf;
    const float *beta = &bet;


    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n,n,n,alpha, A, lda, B, ldb, beta, C, ldc);

    cudaDeviceSynchronize();
}
//Referenced from
//https://solarianprogrammer.com/2012/05/31/matrix-multiplication-cuda-cublas-curand-thrust/
