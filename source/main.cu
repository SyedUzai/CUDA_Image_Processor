#include "image_utils.hpp"
#include "Gaussian.cuh"
using namespace std;


int main() {
    Image instance;
    ifstream file("C:\\Users\\Uzair\\OneDrive\\Documents\\Image_Processor\\images\\input\\lenna.ppm", std::ios::binary);
    if (!file) {
        cerr << "Error: Could not open file\n";
        return 1;
    }

    instance = get_imgData(file);

    size_t imageSize = instance.rawdata.size() * sizeof(unsigned char);
    unsigned char* d_data;


    cudaMalloc((void**)&d_data, imageSize);
    cudaMemcpy(d_data, instance.rawdata.data(), imageSize, cudaMemcpyHostToDevice);

    int threadspblock = 256;
    int blocks = (instance.rawdata.size() + threadspblock - 1) / threadspblock;

    GaussianFilter<<< blocks, threadspblock >>>(d_data, instance.width, instance.height, instance.maxval);

    cudaMemcpy(instance.rawdata.data(), d_data, imageSize, cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    process_pgm(instance, "output.pgm");

    return 0;


}