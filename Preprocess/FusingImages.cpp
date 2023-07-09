//
// Created by Etch on 6/27/2023.
//
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>

int main()
{
    // Directory paths for the three sets of images
    std::string path1 = "D:\\Repo\\UATD\\Dataset\\Processed\\Thorp\\Train\\images512\\";
    std::string path2 = "D:\\Repo\\UATD\\Dataset\\Processed\\Gaussian\\Train\\images512\\";
    std::string path3 = "D:\\Repo\\UATD\\Dataset\\Training\\images512\\";

    // output folder
    std::string output_folder = "D:\\Repo\\UATD\\Dataset\\Processed\\Fusion\\Train\\images\\";


    // Iterate through all the image files
    for (int i = 1; i <= 7600; ++i)
    {
        // print the progress
        std::cout << "Processing image " << i << std::endl;

        // Generate file names
        std::ostringstream str_stream;
        str_stream << std::setw(5) << std::setfill('0') << i;
        std::string filename = str_stream.str();

        // Read images from the three directories
        cv::Mat img1 = cv::imread(path1 + filename + ".bmp", cv::IMREAD_GRAYSCALE);
        cv::Mat img2 = cv::imread(path2 + filename + ".bmp", cv::IMREAD_GRAYSCALE);
        cv::Mat img3 = cv::imread(path3 + filename + ".bmp", cv::IMREAD_GRAYSCALE);

        // Check if the images are loaded
        if (!img1.data || !img2.data || !img3.data)
        {
            std::cout << "Error: Could not open or find the image" << std::endl;
            return -1;
        }

        // Check if all the images have the same size
        if (img1.size() != img2.size() || img1.size() != img3.size())
        {
            std::cout << "Error: Images are not the same size" << std::endl;
            return -1;
        }

        // Merge the three images into one
        cv::Mat img_merged;
        cv::Mat in[] = { img1, img2, img3 };
        cv::merge(in, 3, img_merged);

        // Save the merged image to disk
        cv::imwrite(output_folder + filename + ".bmp", img_merged);
    }

    return 0;
}
