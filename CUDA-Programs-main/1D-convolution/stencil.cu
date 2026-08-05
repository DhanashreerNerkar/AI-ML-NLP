#include <cuda.h>
#include <stdio.h>
#include "stencil.cuh"

__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R){

        int in = threadIdx.x + blockIdx.x * blockDim.x;
        
        extern __shared__ float sdata[]; //Declare shared memory
        int tid = threadIdx.x;

        int dR = (int)R;
                
        float* sharedM_mask = sdata;    
        //Copy mask from host to shared memory             
         if(tid < 2*dR+1){
                sharedM_mask[tid] = mask[tid];
        }

        float* sharedM_output = sharedM_mask+ 2*dR + 1;
        //Initialize output in shared memory to 0
        if(tid<n){
                sharedM_output[tid] = 0;
        }

        float* sharedM_image  =  sharedM_output + blockDim.x + dR;
        //Copy image from host to shared memory
         if(tid<n){
                sharedM_image[tid] = image[in];
        }

        //1D convolution boundary conditions
        if(tid<dR){
            if(in - dR > 0){
                sharedM_image[tid - dR] = image[in - dR];
            }
            else{
                sharedM_image[tid - dR] = 0;
            }
        }
        else if(blockDim.x - tid < dR){
            if(in + dR < n){
                sharedM_image[tid + dR] = image[in + dR]; 
            }
            else{
                sharedM_image[tid + dR] = 0;
            }
        }

        __syncthreads();

        for(int k = -dR; k <= dR; k++){
                sharedM_output[tid] += sharedM_image[tid + k] * sharedM_mask[k+dR];
        }


        output[in] = sharedM_output[tid]; 

}

__host__ void stencil(const float* image,
                      const float* mask,
                      float* output,
                      unsigned int n,
                      unsigned int R,
                      unsigned int threads_per_block){  
                                                        size_t blocks = (((n) + threads_per_block -1)/threads_per_block);
                                                        int sdata_size = ((2*R + threads_per_block) + (2*R+1) + (threads_per_block)) * sizeof(float);
                                                        stencil_kernel<<<blocks,threads_per_block, sdata_size>>>(image, mask, output, n, R);
                      }


            

