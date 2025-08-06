#include <image_utils.hpp>
using namespace std;

void process_pgm(const Image &img, const string &path){
    ofstream out(path, ios::binary);
    out << "P5\n";
    out << img.width << " " << img.height << "\n";
    out << img.maxval << "\n";
    out.write(reinterpret_cast<const char *>(img.rawdata.data()),img.rawdata.size());
}


Image get_imgData(ifstream& file) {
    string line;
    Image  img;
    bool   isRGB;
    
    getline(file, line);
    isRGB = (line == "P6");
   
    do {
        getline(file, line);
    } while (line[0] == '#');

    stringstream dimensions(line);
    dimensions >> img.width >> img.height;

    file >> img.maxval;
    file.ignore(1);

    size_t channels = isRGB ? 3 : 1;
    size_t size = img.width * img.height * channels;
    img.rawdata.resize(size);
    file.read(reinterpret_cast<char *>(img.rawdata.data()), size);

    if (isRGB) {
        vector<unsigned char> gray_data(img.width * img.height);
        for (size_t i = 0, j = 0; i < img.rawdata.size(); i+=3, j++) {
            unsigned char r = img.rawdata[i];
            unsigned char g = img.rawdata[i + 1];
            unsigned char b = img.rawdata[i + 2];

            gray_data[j] = static_cast<unsigned char>(0.299 * r + 0.587 * g + 0.114 * b);
        }
        img.rawdata.resize(img.width * img.height);
        img.rawdata = gray_data;
    }
  
    process_pgm(img, "C:\\Users\\Uzair\\OneDrive\\Documents\\Image_Processor\\images\\output\\output_greyscale.pgm");
    cout << "Width: " << img.width << " Height: " << img.height << " Maxvalue: " << img.maxval << "\n";
    cout << " Size of image is: " << img.rawdata.size() << "\n";
    return img;
}


int main(){
    Image instance;
    ifstream file("C:\\Users\\Uzair\\OneDrive\\Documents\\Image_Processor\\images\\input\\lenna.ppm", std::ios::binary);
    if (!file) {
        cerr << "Error: Could not open file\n";
        return 1;
    }

    instance = get_imgData(file);
    return 0;
   
    
}