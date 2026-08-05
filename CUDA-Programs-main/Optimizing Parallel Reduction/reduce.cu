#include <cuda.h>
#include <stdio.h>
#include "reduce.cuh"

__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n){
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i   = (blockIdx.x*(2*blockDim.x)) + threadIdx.x;
    if (i >= n) {
        sdata[tid] = 0;
    } else {
        sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
    }

    __syncthreads();

    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];

}

__host__ void reduce(float **input, float **output, unsigned int N,
                     unsigned int threads_per_block){
                        

    for (int l = N; l > 1; l = (l + threads_per_block - 1) / (2*threads_per_block)) {
        unsigned int block_num;
        if(l == N){
            block_num = ((l/2) + threads_per_block - 1)/(threads_per_block);
        }
        else{
            block_num = (l + threads_per_block - 1)/(threads_per_block);
        }
        reduce_kernel<<<block_num, threads_per_block, threads_per_block * sizeof(float) >>>(*input, *output, l);

        cudaMemset(*input, 0, N * sizeof(float));

        cudaMemcpy(*input, *output, block_num * sizeof(float), cudaMemcpyDeviceToDevice);

        cudaMemset(*output, 0, N * sizeof(float));

    }

    cudaDeviceSynchronize();

}
