#include <cuda.h>
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <time.h>
#include <chrono>
#include <stdlib.h>
#include "scan.cuh"


int main(int argc, char *argv[]){

    // 
    unsigned int n; unsigned int threads_per_block;
    n = std::atoi(argv[1]); 
    threads_per_block = std::atoi(argv[2]);
    //n = 8; 
    //threads_per_block = 32;


    int size = (n)*sizeof(float);
    float *A; float *result;

    cudaMallocManaged(&A, size);
    cudaMallocManaged(&result, size);

    for(int i=0; i<n; i++){

         A[i]= (float)(rand()) / float(RAND_MAX) * (2) - 1;
        
    }
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    scan1(A,result,n,threads_per_block);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
     
    // Get the elapsed time in milliseconds
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);

    printf("%f ", result[(n)-1]); printf("\n");
    printf("%f ", ms); printf("\n");
    

    
    return 0;
}
