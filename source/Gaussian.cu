
__global__ void GaussianFilter(unsigned char* d_data, unsigned char* d_outdata, int width, int height, int maxval) {
    //Each thread handles one pixel 
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    const int BlurMatrix[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };

    int row = idx / width;
    int col = idx % height;

    if (row < 1 || row >= height - 1 || col < 1 || col >= width - 1) {
        if (row < height && col < width) {
            d_outdata[idx] = d_data[idx]; 
        }
        return;
    }

    int sum = 0;
    
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int neighbourRow = row + dy;
            int neighbourCol = col + dx;
            int pixel = d_data[neighbourRow * width + neighbourCol];
            sum += pixel * BlurMatrix[dy + 1][dx + 1];
        }
    }

    unsigned char blurred = static_cast<unsigned char>(sum / 16);
    d_outdata[row * width + col] = blurred;

    
}
