#include <cuda.h>
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <time.h>
#include <chrono>
#include <stdlib.h>
#include <cublas_v2.h>
#include "mmul.h"

int main(int argc, char *argv[]){

    // 
    std::size_t n; std::size_t n_tests;
    n = std::atoi(argv[1]); 
    n_tests = std::atoi(argv[2]);

    using namespace std;
    int size = (n*n)*sizeof(float);
    float *A; float *B; float *C;

    cudaMallocManaged(&A, size);
    cudaMallocManaged(&B, size);
    cudaMallocManaged(&C, size);
    srand(time(NULL));

    for(int i=0; i<n*n; i++){

        A[i]= (float)(rand()) / float(RAND_MAX) * (2) - 1;
        B[i]= (float)(rand()) / float(RAND_MAX) * (2) - 1;
    }
    
    float time = 0;
    
    for(int i = 0; i < n_tests; i++){
    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    mmul(handle,A,B,C,n);
    // Destroy the handle
    cublasDestroy(handle);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
     
    // Get the elapsed time in milliseconds
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    time = time + ms;

    /*int j = 0;
    while(j < n*n){
        C[j] = 0;
        j++;
    }*/
    }
    
    cout << time/n_tests << endl;
    /*printf("A is  ");
    for(int i = 0; i < n*n; ++i){
        
        printf("%f ", A[i]);
    }
    printf ("\n");
    printf("B is  ");
     for(int i = 0; i < n*n; ++i){
        
        
        printf("%f ", B[i]);
    }
    printf ("\n");
    printf("C is  ");
     for(int i = 0; i < n*n; ++i){
        
        
        printf("%f ", C[i]);
    }*/
    return 0;
}
