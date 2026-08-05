#include <cuda.h>
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <time.h>
#include <chrono>
#include <stdlib.h>
#include "reduce.cuh"

int main(int argc, char *argv[]){

    // 
    std::size_t n; std::size_t threads;
    n = std::atoi(argv[1]); 
    threads = std::atoi(argv[2]);
    //n = 6; 
    //threads = 1024;

    
    srand(time(NULL));
    // Check
   
    int size = (n)*sizeof(float);
    float* result = (float*) malloc((n)*sizeof(float));
    float* A = (float*) malloc((n)*sizeof(float));
    
     for(int i=0; i<n; i++){

        A[i]= (float)(rand()) / float(RAND_MAX) * (2) - 1;
        }
    
    
    float **dA; float **dresult;

    // Allocate space for device
    cudaMalloc((void **)&dA, size);
    cudaMalloc((void **)&dresult, size);

    cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);
    
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    reduce(dA,dresult,n,threads);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    //Copy result from device to host
    cudaMemcpy(result, dresult, size, cudaMemcpyDeviceToHost);

    // Get the elapsed time in milliseconds
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
 
    printf("%f ", result[0]); printf("\n");
    printf("%f ", ms); printf("\n");
    
    /*printf("A is  ");
    for(int i = 0; i < n; ++i){
        
        printf("%f ", A[i]); printf("\n");
    }

    printf("result is  ");
    for(int i = 0; i < n; ++i){
        
        printf("%f ", result[i]); printf("\n");
    }*/
    return 0;
}