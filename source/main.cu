#include "image_utils.hpp"
#include "Gaussian.cuh"
using namespace std;

int main() {
    Image instance;
    ifstream file("C:\\Users\\Uzair\\OneDrive\\Documents\\Image_Processor\\images\\input\\pepper.ppm", std::ios::binary);
    if (!file) {
        cerr << "Error: Could not open file\n";
        return 1;
    }

    instance = get_imgData(file);

    size_t imageSize = instance.rawdata.size() * sizeof(unsigned char);
    unsigned char* d_data;
    unsigned char* d_outdata;
    unsigned char* d_outdata_Sobel;

    cudaMalloc((void**)&d_data, imageSize);
    cudaMalloc((void**)&d_outdata, imageSize);
    cudaMalloc((void**)&d_outdata_Sobel , imageSize);

    cudaMemcpy(d_data, instance.rawdata.data(), imageSize, cudaMemcpyHostToDevice);

    int threadspblock = 256;
    int blockspergrid = (instance.rawdata.size() + threadspblock - 1) / threadspblock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    GaussianFilter<<< blockspergrid, threadspblock >>>(d_data, d_outdata, instance.width, instance.height);
    cudaDeviceSynchronize();

    SobelFilter << < blockspergrid, threadspblock >> > (d_outdata, d_outdata_Sobel, instance.width, instance.height);
    cudaDeviceSynchronize();
    

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU Time: " << milliseconds << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(instance.rawdata.data(), d_outdata, imageSize, cudaMemcpyDeviceToHost);
    process_pgm(instance, "C:\\Users\\Uzair\\OneDrive\\Documents\\Image_Processor\\images\\output\\output_Gaussian.pgm");

    cudaMemcpy(instance.rawdata.data(), d_outdata_Sobel, imageSize, cudaMemcpyDeviceToHost);
    process_pgm(instance, "C:\\Users\\Uzair\\OneDrive\\Documents\\Image_Processor\\images\\output\\output_Gaussian_Sobel.pgm");

    cudaFree(d_data);
    cudaFree(d_outdata);
    cudaFree(d_outdata_Sobel);

    
    return 0;
}