__global__ void SobelFilter(unsigned char* input, unsigned char* output, int width, int height) {
    //Each thread handles one pixel 
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    int totalPixel = width * height;

    if (idx >= totalPixel)
        return;

    int y = idx / width;
    int x = idx % height;

    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        output[y * width + x] = 0;
        return;
    }

    int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int Gy[3][3] = {
        { 1,  2,  1},
        { 0,  0,  0},
        {-1, -2, -1}
    };

    float gradientX = 0.0f;
    float gradientY = 0.0f;

    // Apply the Sobel filter
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int neighborX = x + dx;
            int neighborY = y + dy;
            int neighborIdx = neighborY * width + neighborX;
            unsigned char pixelVal = input[neighborIdx];

            gradientX += pixelVal * Gx[dy + 1][dx + 1];
            gradientY += pixelVal * Gy[dy + 1][dx + 1];
        }
    }

    float magnitude = sqrtf(gradientX * gradientX + gradientY * gradientY);

    // Clamp to [0, 255]
    output[y * width + x] = min(255.0f, magnitude);
}






__global__ void GaussianFilter(unsigned char* d_data, unsigned char* d_outdata, int width, int height) {
    //Each thread handles one pixel 
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    int d_filter[3][3] = {
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
            sum += pixel * d_filter[dy + 1][dx + 1];
        }
    }

    unsigned char blurred = static_cast<unsigned char>(sum / 16);
    d_outdata[row * width + col] = blurred;

    
}
