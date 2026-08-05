#include <cuda.h>
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <time.h>
#include <chrono>
#include <stdlib.h>
#include <cub/cub.cuh>
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>
#include "cub/util_debug.cuh"

using namespace cub;
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

int main(int argc, char** argv) {
    std::size_t n; n = atol(argv[1]);
    float *h_in = new float[n]; float result = 0;
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        h_in[i] = (float)(rand()) / float(RAND_MAX) * (2) - 1;
    }

    float* d_in = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)& d_in, sizeof(float) * n));
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(float) * n, cudaMemcpyHostToDevice));
    float* d_sum = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)& d_sum, sizeof(float) * 1));
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, n));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, n));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

   
    CubDebugExit(cudaMemcpy(&result, d_sum, sizeof(int) * 1, cudaMemcpyDeviceToHost));

    printf("%f ", result); printf("\n");
    printf("%f ", ms); printf("\n");

    /*printf("A is  "); printf("\n"); float check = 0;
    for(int i = 0; i < n; ++i){
        
        printf("%f ", h_in[i]); printf("\n");
        check = check + h_in[i];

    }
    printf("Correct answer is  ");
    printf("%f ", check); printf("\n");*/
    // Cleanup
    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
    if (d_sum) CubDebugExit(g_allocator.DeviceFree(d_sum));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
    return 0;

}