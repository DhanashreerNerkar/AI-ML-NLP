#include "scan.cuh"
#include <cuda.h>
#include <stdio.h>

__global__ void scan_kernel(const float* input, float* output, int n) {
    extern volatile __shared__  float temp[]; // allocated on invocation
    float f = 0; 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    // Copying data from global to shared memory
    if (index < n) {
        temp[tid] = input[index];
    }
    else {
        temp[tid] = 0;
    }
    __syncthreads();

    for(int step = 1; step <= blockDim.x; step *= 2) 
    {
        if(tid >= step)
        {
            f = temp[tid-step];
            temp[tid] += f;
        }
        __syncthreads();
    }
    __syncthreads();
    if (index < n)
        output[index] = temp[tid];  
    
}

__global__ void extra(const float* input, float* output, unsigned int num_blocks, unsigned int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    
    for(int step = 1; step <= num_blocks; step *= 2) 
    {
        unsigned int start_block_idx = blockIdx.x * blockDim.x - (step - 1) * blockDim.x - 1;
        if(bid >= step && index < n)
        {
            output[index] += output[start_block_idx];
        }
        __syncthreads();
    }
    
}


__host__ void scan(const float* input, float* output, unsigned int n, unsigned int threads_per_block){
    unsigned int blocks = ((((n) + threads_per_block -1)/threads_per_block));
    unsigned int sdata_size = 2*threads_per_block;
    scan_kernel<<<blocks,threads_per_block, sdata_size>>>(input,output, n);
   
    extra <<< blocks, threads_per_block >>> (input,output,blocks, n);
    
}
