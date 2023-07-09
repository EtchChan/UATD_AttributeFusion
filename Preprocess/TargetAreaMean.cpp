//
// Created by Etch on 6/14/2023.
//
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <opencv2/opencv.hpp>

std::vector<std::string> readLines(std::string filename)
{
    std::ifstream infile(filename);
    std::string line;
    std::vector<std::string> lines;
    while (std::getline(infile, line))
    {
        lines.push_back(line);
    }
    return lines;
}

int main()
{
    // Define frequencies
    std::vector<int> frequencies = {1200, 720};
    std::map<int, std::ofstream> freq_to_file;

    // Open files for each frequency
    for(auto freq: frequencies)
    {
        std::string filename = "D:\\Repo\\UATD\\Dataset\\Training\\target_average_"
                + std::to_string(freq) + ".txt";
        freq_to_file[freq].open(filename, std::ios_base::app);
    }

    // Read range_frequency.txt
    auto range_frequency = readLines("D:\\Repo\\UATD\\Dataset\\Training\\range_frequency.txt");

    for(int i = 1; i <= 7600; ++i)
    {
        // print progress
        if(i % 100 == 0)
        {
            std::cout << "Processing image " << i << std::endl;
        }
        std::ostringstream img_file;
        img_file << "D:\\Repo\\UATD\\Dataset\\Training\\images\\" <<
        std::setw(5) << std::setfill('0') << i << ".bmp";
        std::ostringstream txt_file;
        txt_file << "D:\\Repo\\UATD\\Dataset\\Processed\\Thorp\\Train\\annotations\\" <<
        std::setw(5) << std::setfill('0') << i << ".txt";

        cv::Mat img = cv::imread(img_file.str(), cv::IMREAD_GRAYSCALE);

        // Check if image was successfully read
        if(img.empty())
        {
            std::cout << "Could not read image: " << img_file.str() << std::endl;
            continue;
        }

        // Parse the txt file
        auto bndboxes = readLines(txt_file.str());
        for(int j = 1; j < bndboxes.size(); ++j)
        {
            std::istringstream iss(bndboxes[j]);
            int name, xmin, ymin, xmax, ymax;
            if(!(iss >> name >> xmin >> ymin >> xmax >> ymax))
            {
                break;
            }

            // Calculate the averages

            cv::Rect roi(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
//            if(0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= img.cols && 0 <= roi.y && 0 <= roi.height && roi.y + roi.height <= img.rows)
//            {
//                // Do nothing
//            }
//            else
//            {
//                std::cout << "Bounding box out of bounds: " << txt_file.str() << std::endl;
//                system("pause");
//            }
            cv::Mat obj_img = img(roi);
            double area_average = cv::mean(obj_img)[0];

            cv::Range row_range(ymin, ymax + 1);
            cv::Range col_range(0, img.cols);
            cv::Mat row_img = img(row_range, col_range);

            double row_average = cv::mean(row_img)[0];

            double ratio = area_average / row_average;

            // Write the averages to the corresponding file
            int freq = std::stoi(range_frequency[i-1].substr(range_frequency[i-1].find(' ')+1));
            freq_to_file[freq] << area_average << " " << row_average << " " <<  ratio << "\n";
        }
    }

    // Close files
    for(auto freq: frequencies)
    {
        freq_to_file[freq].close();
    }

    return 0;
}