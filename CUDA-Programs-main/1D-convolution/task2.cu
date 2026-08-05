#include <cuda.h>
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <time.h>
#include <chrono>
#include <stdlib.h>
#include "stencil.cuh"

int main(int argc, char *argv[]){

    // 
    std::size_t n; std::size_t threads; std::size_t R;
    n = std::atoi(argv[1]); 
    R = std::atoi(argv[2]);
    threads = std::atoi(argv[3]);
    //n = 4;

    srand(time(NULL));

    int size = (n)*sizeof(float);
    float* output = (float*) malloc((n)*sizeof(int));
    float* image = (float*) malloc((n)*sizeof(int));
    float* mask = (float*) malloc((2*R+1)*sizeof(int));

    for(int i=0; i<n; i++){

        image[i]= (float)(rand()) / float(RAND_MAX) * (2) - 1;
        
     }

    for(int i=0; i<(2*R+1); i++){

        mask[i]= (float)(rand()) / float(RAND_MAX) * (2) - 1;
    }

     float *dimage; float *dmask; float *doutput;

      // Allocate space for device
    cudaMalloc((void **)&dimage, size);
    cudaMalloc((void **)&dmask, size);
    cudaMalloc((void **)&doutput, size);

    cudaMemcpy(dimage, image, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dmask, mask, size, cudaMemcpyHostToDevice);

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    stencil(dimage,dmask,doutput,n, R, threads);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    //Copy result from device to host
    cudaMemcpy(output, doutput, size, cudaMemcpyDeviceToHost);

    // Get the elapsed time in milliseconds
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
 
    printf("%f ", output[n-1]); printf("\n");
    printf("%f ", ms); printf("\n");


return 0;
}
