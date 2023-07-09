//
// Created by Etch on 6/13/2023.
//
#include <opencv2/opencv.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <unordered_map>

int main() {
    std::ifstream rf("D:\\Repo\\UATD\\Dataset\\Training\\range_frequency.txt");
    std::string inputFolder = "D:\\Repo\\UATD\\Dataset\\Training\\images\\";
    std::string outputFolder = "D:\\Repo\\UATD\\Dataset\\Training\\";
    if (!rf) {
        std::cerr << "Unable to open range_frequency.txt" << std::endl;
        return 1;
    }

    std::unordered_map<int, std::ofstream> file_map;

    std::string line;
    int idx = 1;

    while (std::getline(rf, line)) {
        std::istringstream iss(line);
        double range;
        unsigned int frequency;
        if (!(iss >> range >> frequency)) {
            std::cerr << "Invalid format in range_frequency.txt" << std::endl;
            return 1;
        }

        std::ostringstream img_name_stream;
        img_name_stream << inputFolder << std::setw(5) << std::setfill('0') << idx << ".bmp";
        cv::Mat img = cv::imread(img_name_stream.str(), cv::IMREAD_GRAYSCALE);

        if (img.empty()) {
            std::cerr << "Could not open or find the image " << img_name_stream.str() << ".bmp" << std::endl;
            continue;
        }

        if (file_map.find(frequency) == file_map.end()) {
            std::ofstream ofs(outputFolder + "Attenuation_data_" + std::to_string((int)frequency) + ".txt");
            file_map[frequency] = std::move(ofs);
        }
        std::ofstream &ofs = file_map[frequency];

        // using the average of the first row to normalize the average value of the left rows
        cv::Scalar first_row_mean = cv::mean(img.row(0));

        for (int i = 0; i < img.rows; ++i) {
            cv::Scalar row_mean = cv::mean(img.row(i));
            // normalize the value by the average of the first row
            row_mean[0] /= first_row_mean[0];
            double distance = i / (double)img.rows * range;
            ofs << distance << " " << row_mean[0] << std::endl;
        }

        ++idx;
    }

    for (auto &pair : file_map) {
        pair.second.close();
    }

    return 0;
}
