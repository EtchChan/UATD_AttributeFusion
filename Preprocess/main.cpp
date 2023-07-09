#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <limits>
#include <numeric>


//using namespace std;
//using namespace cv;



struct Data {
    double area_average;
    double row_average;
    double ratio;
};

struct Stats {
    double min = std::numeric_limits<double>::max();
    double max = std::numeric_limits<double>::min();
    double mean = 0;
};

void computeStats(const std::string &filename) {
    std::ifstream file(filename);
    std::vector<Data> data;
    Data temp;
    Stats areaStats, rowStats, ratioStats;
    double area_stddev = 0, row_stddev = 0, ratio_stddev = 0;

    while (file >> temp.area_average >> temp.row_average >> temp.ratio) {
        data.push_back(temp);
        areaStats.min = std::min(areaStats.min, temp.area_average);
        areaStats.max = std::max(areaStats.max, temp.area_average);

        rowStats.min = std::min(rowStats.min, temp.row_average);
        rowStats.max = std::max(rowStats.max, temp.row_average);

        ratioStats.min = std::min(ratioStats.min, temp.ratio);
        ratioStats.max = std::max(ratioStats.max, temp.ratio);
    }

    areaStats.mean = std::accumulate(data.begin(), data.end(), 0.0, [](double sum, const Data &d) { return sum + d.area_average; }) / data.size();
    rowStats.mean = std::accumulate(data.begin(), data.end(), 0.0, [](double sum, const Data &d) { return sum + d.row_average; }) / data.size();
    ratioStats.mean = std::accumulate(data.begin(), data.end(), 0.0, [](double sum, const Data &d) { return sum + d.ratio; }) / data.size();

    for (const auto &d : data) {
        area_stddev += (d.area_average - areaStats.mean) * (d.area_average - areaStats.mean);
        row_stddev += (d.row_average - rowStats.mean) * (d.row_average - rowStats.mean);
        ratio_stddev += (d.ratio - ratioStats.mean) * (d.ratio - ratioStats.mean);
    }

    area_stddev = std::sqrt(area_stddev / data.size());
    row_stddev = std::sqrt(row_stddev / data.size());
    ratio_stddev = std::sqrt(ratio_stddev / data.size());

    std::cout << filename << " stats:" << std::endl;
    std::cout << "Area Average: Min = " << areaStats.min << ", Max = " << areaStats.max << ", Mean = " << areaStats.mean << ", stddev = " << area_stddev << std::endl;
    std::cout << "Row Average: Min = " << rowStats.min << ", Max = " << rowStats.max << ", Mean = " << rowStats.mean << ", stddev = " << row_stddev << std::endl;
    std::cout << "Ratio: Min = " << ratioStats.min << ", Max = " << ratioStats.max << ", Mean = " << ratioStats.mean << ", stddev = " << ratio_stddev << std::endl;
}

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

void get_YOLO_labels(const std::string &foldername)
{
    for(int i = 1; i <= 800; ++i)
    {
        std::ostringstream annotation_file;
        std::ostringstream label_file;
        annotation_file << foldername << "annotations\\" << std::setw(5) << std::setfill('0') << i << ".txt";
        label_file << foldername << "labels\\" << std::setw(5) << std::setfill('0') << i << ".txt";

        // Parse the annotation file
        auto bndboxes = readLines(annotation_file.str());
        // get image width, height, and elevation
        int width, height;
        std::string elevation;
        std::istringstream image_info(bndboxes[0]);
        image_info >> width >> height >> elevation;
        // get bounding boxes locations
        for(int j = 1; j < bndboxes.size(); ++j)
        {
            std::istringstream iss(bndboxes[j]);
            int name, xmin, ymin, xmax, ymax;
            if(!(iss >> name >> xmin >> ymin >> xmax >> ymax))
            {
                break;
            }

            // compute the center of the bounding box
            int x_center = (xmin + xmax) / 2;
            int y_center = (ymin + ymax) / 2;
            // compute the width and height of the bounding box
            int box_width = xmax - xmin + 1;
            int box_height = ymax - ymin + 1;
            // normalize the center and size of the bounding box
            double x_center_norm = (double)x_center / width;
            double y_center_norm = (double)y_center / height;
            double box_width_norm = (double)box_width / width;
            double box_height_norm = (double)box_height / height;
            // write the label to the label file
            std::ofstream label(label_file.str(), std::ios_base::app);
            label << name << " " << x_center_norm << " " << y_center_norm << " " << box_width_norm << " " << box_height_norm << '\n';
        }

    }
}

// Function to resize the images to 512 * 512
void resize_images(const std::string &foldername)
{
    for(int i = 1; i <= 800; ++i)
    {
        std::ostringstream input_file;
        std::ostringstream output_file;
        input_file << foldername << "images\\" << std::setw(5) << std::setfill('0') << i << ".bmp";
        output_file << foldername << "images512\\" << std::setw(5) << std::setfill('0') << i << ".bmp";

        // Read image
        cv::Mat image = cv::imread(input_file.str(), cv::IMREAD_GRAYSCALE);

        // Check for failure
        if(image.empty())
        {
            std::cout << "Could not open or find the image " << i << std::endl;
        }
        else
        {
            std::cout << "Image loaded successfully " << i << std::endl;
        }

        // Resize the image
        cv::Mat resized_image;
        cv::resize(image, resized_image, cv::Size(512, 512));

        // Write the resized image
        cv::imwrite(output_file.str(), resized_image);
    }
}

// Function to Normalize the original images with min/max normalization
void normalize_images(const std::string &foldername)
{
    for(int i = 1; i <= 800; ++i)
    {
        std::ostringstream input_file;
        input_file << foldername << "images\\" << std::setw(5) << std::setfill('0') << i << ".bmp";

        // Read image
        cv::Mat image = cv::imread(input_file.str(), cv::IMREAD_GRAYSCALE);

        // Check for failure
        if(image.empty())
        {
            std::cout << "Could not open or find the image " << i << std::endl;
        }
        else
        {
            std::cout << "Image loaded successfully " << i << std::endl;
        }

        // Normalize the image
        cv::Mat normalized_image;
        cv::normalize(image, normalized_image, 0, 255, cv::NORM_MINMAX);

        // Write the normalized image
        cv::imwrite(input_file.str(), normalized_image);
    }
}


int main() {
    //********** Test of image read and show **********//
//    // Read image
//    Mat image = imread("D:\\Repo\\UATD\\Dataset\\Training\\images\\00001.bmp", IMREAD_COLOR);
//
//    // Check for failure
//    if(image.empty())
//    {
//        cout << "Could not open or find the image" << endl;
//    }
//    else
//    {
//        cout << "Image loaded successfully" << endl;
//    }
//
//    // Display image in a window
//    namedWindow("Display window", WINDOW_AUTOSIZE);
//    imshow("Display window", image);
//
//    // keep window open until user presses a key
//    waitKey(0);
//    destroyAllWindows();




    //********** Find the minimum width and height of images **********//
//    int min_width = INT_MAX;
//    int min_height = INT_MAX;
//    for(int i = 1; i <= 7600; ++i) {
//        string inputFolder = "D:\\Repo\\UATD\\Dataset\\Training\\images\\";
//
//        ostringstream inputFilename;
//        inputFilename << inputFolder << setw(5) << setfill('0') << i << ".bmp";
//
//        cv::Mat img = cv::imread(inputFilename.str(), cv::IMREAD_UNCHANGED);
//        if(img.empty()) {
//            std::cout << "Could not open or find the image " << inputFilename.str() << std::endl;
//            return -1;
//        }
//
//        if(img.cols < min_width) {
//            min_width = img.cols;
//        }
//        if(img.rows < min_height) {
//            min_height = img.rows;
//        }
//    }
//
//    cout << "Minimum width: " << min_width << endl;
//    cout << "Minimum height: " << min_height << endl;
//
//    system("pause");


    //********** Find min, max, mean of Target Area **********//
//    computeStats("D:\\Repo\\UATD\\Dataset\\Training\\target_average_720.txt");
//    computeStats("D:\\Repo\\UATD\\Dataset\\Training\\target_average_1200.txt");

    //********** convert annotations to YOLO labels **********//
//    get_YOLO_labels("D:\\Repo\\UATD\\Dataset\\Processed\\Thorp\\Test\\");

    //********** resize images to 512 * 512 **********//
    resize_images("D:\\Repo\\UATD\\Dataset\\Processed\\Gaussian\\Validate\\");

    //********** normalize original images for further fusion **********//
//    normalize_images("D:\\Repo\\UATD\\Dataset\\Test\\");

    return 0;
}
