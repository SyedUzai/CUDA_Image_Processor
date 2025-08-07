#include "image_utils.hpp"
#include <chrono>
using namespace std;

void Gaussianfilter_cpu(unsigned char* d_data, unsigned char* d_outdata, int width, int height, int maxval) {
    const int BlurMatrix[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };

    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            // Handle border pixels by copying them directly
            if (row < 1 || row >= height - 1 || col < 1 || col >= width - 1) {
                d_outdata[row * width + col] = d_data[row * width + col];
                continue;
            }

            int sum = 0;

            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int neighborRow = row + dy;
                    int neighborCol = col + dx;
                    int pixel = d_data[neighborRow * width + neighborCol];
                    sum += pixel * BlurMatrix[dy + 1][dx + 1];
                }
            }

            unsigned char blurred = static_cast<unsigned char>(sum / 16);
            d_outdata[row * width + col] = blurred;
        }
    }
}

int main() {
    Image instance;
    ifstream file("C:\\Users\\Uzair\\OneDrive\\Documents\\Image_Processor\\images\\input\\pepper.ppm", std::ios::binary);
    if (!file) {
        cerr << "Error: Could not open file\n";
        return 1;
    }

    instance = get_imgData(file);
    size_t imageSize = instance.rawdata.size() * sizeof(unsigned char);

    unsigned char* d_data = new unsigned char[imageSize];
    unsigned char* d_outdata = new unsigned char[imageSize];

    
    memcpy(d_data, instance.rawdata.data(), imageSize);

    auto start = std::chrono::high_resolution_clock::now();

    Gaussianfilter_cpu(d_data, d_outdata, instance.width, instance.height, instance.maxval);

    auto stop = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = stop - start;

    std::cout << "CPU Time: " << duration.count() << " ms\n";

    memcpy(instance.rawdata.data(), d_outdata, imageSize);

    process_pgm(instance, "C:\\Users\\Uzair\\OneDrive\\Documents\\Image_Processor\\images\\output\\output_Gaussian.pgm");

    
    delete[] d_data;
    delete[] d_outdata;

    return 0;
}
