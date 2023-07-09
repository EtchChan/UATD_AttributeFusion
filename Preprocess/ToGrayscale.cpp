//
// Created by Etch on 6/13/2023.
//
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;


void convertToGrayscale(const string& inputPath, const string& outputPath) {
    // Read image
    Mat image = imread(inputPath, IMREAD_COLOR);

    // Check for failure
    if (image.empty()) {
        cerr << "Error: Cannot read image: " << inputPath << endl;
        return;
    }

    // Split image into 3 channels
    vector<Mat> channels;
    split(image, channels);

    // Convert to grayscale, using average of 3 channels as intensity
    Mat grayImage = (channels[0] + channels[1] + channels[2]) / 3;

    if (!imwrite(outputPath, grayImage)) {
        cerr << "Error: Cannot write image: " << outputPath << endl;
    }
}

int main() {
    string inputFolder = "D:\\Repo\\UATD\\Dataset\\Validation\\images\\";
    string outputFolder = "D:\\Repo\\UATD\\Dataset\\Validation\\images\\";
    int startImageIndex = 1;
    int endImageIndex = 800;

    for (int i = startImageIndex; i <= endImageIndex; i++) {
        stringstream inputPath, outputPath;
        inputPath << inputFolder << setfill('0') << setw(5) << i << ".bmp";
        outputPath << outputFolder << setfill('0') << setw(5) << i << ".bmp";

        cout << "Processing image " << i << '\n';
        convertToGrayscale(inputPath.str(), outputPath.str());
    }

    return 0;
}