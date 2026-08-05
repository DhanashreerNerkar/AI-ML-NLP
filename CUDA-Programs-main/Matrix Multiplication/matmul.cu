#include <cuda.h>
#include <stdio.h>
#include "matmul.cuh"

__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n){
        
        int in = threadIdx.x + blockIdx.x * blockDim.x;
        if (in < n*n){
        float value = 0;
        
        for(int k = 0; k<n; k++){
                int rows  = in/n;
                int col = in%n;
                float first = A[rows*n+k];
                float second = B[col + k*n];
                value += first*second; 
                
        }
        C[in] = value;
        }
}

void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block){
        size_t blocks = (((n*n) + threads_per_block -1)/threads_per_block);
        matmul_kernel<<<blocks,threads_per_block>>>(A, B, C, n);
}
