#include <iostream>
#include <image_utils.hpp>

__global void GaussianFilter(unsigned char* d_data, int width, int height, int maxval) {
    return 1;


}

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
    

    cudaMemcpy(instance.rawdata.data(), d_data, imageSize, cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    process_pgm(instance, "output.pgm");

    



    return 0;


}