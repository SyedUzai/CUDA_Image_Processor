#ifndef IMAGE_UTILS_HPP
#define IMAGE_UTILS_HPP

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

struct Image {
    int width;
    int height;
    int maxval;
    std::vector<unsigned char> rawdata;
};

// Writes a grayscale image in PGM (P5) format to the specified file path
void process_pgm(const Image& img, const std::string& path);

// Reads a PGM (P5) or PPM (P6) image, converts to grayscale if RGB, and returns the Image struct
Image get_imgData(std::ifstream& file);

#endif // IMAGE_UTILS_HPP
