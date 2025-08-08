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

    cudaMalloc((void**)&d_data, imageSize);
    cudaMalloc((void**)&d_outdata, imageSize);

    cudaMemcpy(d_data, instance.rawdata.data(), imageSize, cudaMemcpyHostToDevice);

    int threadspblock = 256;
    int blockspergrid = (instance.rawdata.size() + threadspblock - 1) / threadspblock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    GaussianFilter<<< blockspergrid, threadspblock >>>(d_data, d_outdata, instance.width, instance.height);

    cudaDeviceSynchronize();  

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU Time: " << milliseconds << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(instance.rawdata.data(), d_outdata, imageSize, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    cudaFree(d_outdata);

    process_pgm(instance, "C:\\Users\\Uzair\\OneDrive\\Documents\\Image_Processor\\images\\output\\output_Gaussian.pgm");

    return 0;


}