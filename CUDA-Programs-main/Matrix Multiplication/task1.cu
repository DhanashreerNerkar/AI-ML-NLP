#include <cuda.h>
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <time.h>
#include <chrono>
#include <stdlib.h>
#include "matmul.cuh"

int main(int argc, char *argv[]){

    // 
    std::size_t n; std::size_t threads;
    n = std::atoi(argv[1]); 
    threads = std::atoi(argv[2]);
    //n = 4;
    
    //float A[n*n];
	//float B[n*n]; 
    //float result[n*n];
    srand(time(NULL));
    // Check
    /*
    float A[n*n] = {5,2,6,1,0,6,2,0,3,8,1,4,1,8,5,6};
    float B[n*n] = {7,5,8,0,1,8,2,6,9,4,3,8,5,3,7,9};*/

    int size = (n*n)*sizeof(float);
    float* result = (float*) malloc((n*n)*sizeof(int));
    float* A = (float*) malloc((n*n)*sizeof(int));
    float* B = (float*) malloc((n*n)*sizeof(int));

     for(int i=0; i<n*n; i++){

        A[i]= (float)(rand()) / float(RAND_MAX) * (2) - 1;
        B[i]= (float)(rand()) / float(RAND_MAX) * (2) - 1;
     }

     float *dA; float *dB; float *dC;

    // Allocate space for device
    cudaMalloc((void **)&dA, size);
    cudaMalloc((void **)&dB, size);
    cudaMalloc((void **)&dC, size);

    cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice);

    

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul(dA,dB,dC,n,threads);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    //Copy result from device to host
    cudaMemcpy(result, dC, size, cudaMemcpyDeviceToHost);

    // Get the elapsed time in milliseconds
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
 
    printf("%f ", result[(n*n)-1]); printf("\n");
    printf("%f ", ms); printf("\n");
   
}
