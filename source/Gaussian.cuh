#ifndef GAUSSIAN_CUH
#define GAUSSIAN_CUH

// Any includes needed by the kernel
#include <cuda_runtime.h>

// Declare your kernel
__global__ void GaussianFilter(unsigned char* data, int width, int height, int maxval);

#endif // GAUSSIAN_CUH