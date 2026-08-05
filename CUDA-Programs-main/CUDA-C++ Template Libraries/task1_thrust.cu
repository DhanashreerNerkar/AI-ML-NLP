#include <cuda.h>
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <time.h>
#include <chrono>
#include <stdlib.h>
#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>

int main(int argc, char *argv[]){

    std::size_t n;
    n = std::atoi(argv[1]); 
    thrust::host_vector<float> h_vec(n);
    srand(time(NULL));
    for(int i=0; i<n; i++){

         h_vec[i]= (float)(rand()) / float(RAND_MAX) * (2) - 1;
        
    }
    thrust::device_vector<float> d_vec = h_vec;

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    float result = thrust::reduce(d_vec.begin(), d_vec.end());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Get the elapsed time in milliseconds
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);

    printf("%f ", result); printf("\n");
    printf("%f ", ms); printf("\n");
    
    //Check
    /*printf("A is  "); printf("\n"); float check = 0;
    for(int i = 0; i < n; ++i){
        
        printf("%f ", h_vec[i]); printf("\n");
        check = check + h_vec[i];

    }
    printf("Correct answer is  ");
    printf("%f ", check); printf("\n");*/
    
    return 0;
}





