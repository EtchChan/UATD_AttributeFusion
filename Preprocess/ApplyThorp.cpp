//
// Created by Etch on 6/14/2023.
//
#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <string>

struct RangeFrequency {
    double range;
    int frequency;
};

double getAlpha(int frequency) {
    // simple average of square: 1.139314, (average): 0.637352, geometric mean: 0.737402
    if (frequency == 1200) return 0.270824; // using the fitted parameters directly
    // simple average of square: 0.525614, (average): 0.315477, geometric mean: 0.421414
    if (frequency == 720) return 0.211480;
    return 0.0;  // default case
}

double getK(int frequency) {
    // simple average of square: -0.086893
    if (frequency == 1200) return -0.173785; // using the fitted parameters directly
    // simple average of square: -0.122658
    if (frequency == 720) return -0.245315;
    return 0.0;  // default case
}

int main() {
    std::vector<RangeFrequency> rangeFrequencyData;
    std::string line;
    std::ifstream file("D:\\Repo\\UATD\\Dataset\\Test\\range_frequency.txt");
    std::string inputFolder = "D:\\Repo\\UATD\\Dataset\\Test\\images\\";
    std::string outputFolder = "D:\\Repo\\UATD\\Dataset\\Processed\\Thorp\\Test\\images\\";

    if (file.is_open()) {
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            double range;
            int frequency;
            ss >> range >> frequency;
            rangeFrequencyData.push_back({range, frequency});
        }
        file.close();
    } else {
        std::cout << "Unable to open file";
        return 1;
    }

    for (int i = 0; i < rangeFrequencyData.size(); ++i) {
        // print progress
        std::cout << "Processing image " << (i + 1) << " of " << rangeFrequencyData.size() << std::endl;

        // construct filename
        std::ostringstream input_filename;
        std::ostringstream output_filename;
        input_filename << inputFolder << std::setw(5) << std::setfill('0') << (i + 1) << ".bmp";
        output_filename << outputFolder << std::setw(5) << std::setfill('0') << (i + 1) << ".bmp";
        std::string filename = std::to_string(i + 1);
        while (filename.length() < 5) filename = "0" + filename;
        filename = filename + ".bmp";

        // read image
        cv::Mat img = cv::imread(input_filename.str(), cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cout << "Could not open or find the image" << std::endl;
            return -1;
        }

        double alpha = getAlpha(rangeFrequencyData[i].frequency);
        double k = getK(rangeFrequencyData[i].frequency);

        cv::Mat img_double;
        img.convertTo(img_double, CV_64F); // convert image to double precision

        double min_val = DBL_MAX;
        double max_val = DBL_MIN;

        // apply thorp function to each row and find min and max
        for (int r = 0; r < img_double.rows; ++r) {
            double d = static_cast<double>(r) / img_double.rows * rangeFrequencyData[i].range;
            for (int c = 0; c < img_double.cols; ++c) {
                double intensity = img_double.at<double>(r, c);  // normalize to [0, 1]
                intensity = intensity * exp(d * alpha) * pow((d / 0.1 + 1), k);
                img_double.at<double>(r, c) = intensity;

                // update min and max
                if (intensity < min_val) min_val = intensity;
                if (intensity > max_val) max_val = intensity;
            }
        }

        // normalize back to [0, 255] using min and max
        for (int r = 0; r < img_double.rows; ++r) {
            for (int c = 0; c < img_double.cols; ++c) {
                double intensity = (img_double.at<double>( r, c) - min_val) / (max_val - min_val);  // normalize to [0, 1]
                img.at<uchar>(r, c) = static_cast<uchar>(intensity * 255);  // normalize back to [0, 255]
            }
        }

        // write back the image
        if (!cv::imwrite(output_filename.str(), img)) {
            std::cout << "Failed to save the image" << std::endl;
            return -1;
        }
    }

    return 0;
}
