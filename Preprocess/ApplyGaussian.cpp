//
// Created by 30233 on 6/15/2023.
//
/********* include necessary header files from std lib*********/
#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>




/********* include necessary customized header files *********/



/********* definition of local functions *********/
// loads the erf lookup table into a vector(step 0.01 is implicit)
std::vector<double> load_erf_lookup_table(const std::string &filename);

// function that loads frequency info into a vector
std::vector<int> load_frequency_info(const std::string &filename);

// Function to normalize the input value according to its mean and standard deviation
inline double normalize_value(double value, double mean, double std_dev);

// Function to get the Gaussian function value
inline double get_Gaussian(double normalized_value);

// Function to get the skewness
double get_skewness(double normalized_value, double alpha, std::vector<double> &lookup_table);

// Function to merge three coefficients
inline double merge_coefficients(double row_coefficient, double pixel_coefficient, double ratio_coefficient);



/********* definition of local data structures *********/
struct Norm_Param{
    double mean;
    double std_dev;
    double alpha;
};



/********* main function *********/
int main() {
    std::string image_folder = "D:\\Repo\\UATD\\Dataset\\Training\\images\\";
    std::string output_folder = "D:\\Repo\\UATD\\Dataset\\Processed\\Gaussian\\Train\\images\\";
    std::string erf_table_file = "D:\\Repo\\UATD\\Dataset\\Processed\\Thorp\\Train\\erf_lookup_table.txt";
    std::string frequency_file = "D:\\Repo\\UATD\\Dataset\\Training\\range_frequency.txt";

    // Load the erf lookup table and frequency data
    std::vector<double> erf_lookup_table = load_erf_lookup_table(erf_table_file);
    std::vector<int> frequency_info = load_frequency_info(frequency_file);

    // Set the parameters for normalization
    std::unordered_map<int, std::vector<Norm_Param>> norm_params;
    norm_params[720] = {{5.461813, 6.044201, 0.1477960},
                         {1.789798, 2.375928, 4.109044},
                         {2.137244, 1.142129, 0.1488375}};
    norm_params[1200] = {{3.668178, 10.50489, 4.436865},
                        {3.260787, 1.207428, 0.878145},
                        {2.523499, 1.020649, 0.1636565}};

    // Loop over images, specify the range of images 800/7600
    for (int image_index = 1160; image_index <= 7600; image_index++) {
        // print the progress
        std::cout << "Processing image " << image_index << std::endl;

        std::ostringstream img_file;
        std::ostringstream res_file;
        img_file << image_folder << std::setw(5) << std::setfill('0') << image_index << ".bmp";
        res_file << output_folder << std::setw(5) << std::setfill('0') << image_index << ".bmp";

        // read the image
        cv::Mat img = cv::imread(img_file.str(), cv::IMREAD_GRAYSCALE);

        // Check if image was successfully read
        if (img.empty()) {
            std::cout << "Could not read the image: " << img_file.str() << std::endl;
            return 1;
        }

        // Get frequency info for the image
        int frequency = frequency_info[image_index - 1];

        // Create a copy of the image for the result
        cv::Mat res = img.clone();
        res.convertTo(res, CV_64F);

        // Loop over each pixel
        for (int x = 0; x < img.rows; x++) {
            // calculate the average intensity of the row
            cv::Range row_range(x, x + 1);
            cv::Range col_range(0, img.cols);
            cv::Mat row_img = img(row_range, col_range);
            double row_average = cv::mean(row_img)[0];

            // Get the normalization parameters
            double normalized_row_average = normalize_value(row_average, norm_params[frequency][1].mean, norm_params[frequency][1].std_dev);
            // Get the row coefficient
            double row_coefficient = get_Gaussian(normalized_row_average) * get_skewness(normalized_row_average, norm_params[frequency][1].alpha, erf_lookup_table);

            for (int y = 0; y < img.cols; y++) {
                // get the intensity of the pixel and normalize it
                double pixel_intensity = static_cast<double>(img.at<uchar>(x, y));
                double normalized_pixel_intensity = normalize_value(pixel_intensity, norm_params[frequency][0].mean, norm_params[frequency][0].std_dev);
                // Get the pixel coefficient
                double pixel_coefficient = get_Gaussian(normalized_pixel_intensity) * get_skewness(normalized_pixel_intensity, norm_params[frequency][0].alpha, erf_lookup_table);

                // Get ratio pixel_intensity / row_average and normalize it
                double ratio = pixel_intensity / (row_average + 0.00000001); // add a bias to avoid division by zero
                double normalized_ratio = normalize_value(ratio, norm_params[frequency][2].mean, norm_params[frequency][2].std_dev);
                // Get the ratio coefficient
                double ratio_coefficient = get_Gaussian(normalized_ratio) * get_skewness(normalized_ratio, norm_params[frequency][2].alpha, erf_lookup_table);

                // Merge the coefficients and the final coefficient
                double coefficient = merge_coefficients(row_coefficient, pixel_coefficient, ratio_coefficient);

                // Update the result
                res.at<double>(x, y) = coefficient * pixel_intensity;
            }
        }

        // Normalize the results
        cv::normalize(res, res, 0, 255, cv::NORM_MINMAX);

        // Convert back to CV_8U
        res.convertTo(res, CV_8U);

        // Save the result, (replacing the input image)
        cv::imwrite(res_file.str(), res);
    }

    return 0;
}



/********* Implementation of local functions *********/
// Function to load the erf lookup table
std::vector<double> load_erf_lookup_table(const std::string &filename)
{
    // Open the file.
    std::ifstream file(filename);

    // Check if the file opened successfully.
    if (!file) {
        // Throw an exception if the file could not be opened.
        throw std::runtime_error("Could not open file!");
    }

    // Create a vector to store the y values.
    std::vector<double> y_values;

    // Read each line from the file.
    for (std::string line; std::getline(file, line);) {
        // Split the line into two parts, the x value and the y value.
        std::stringstream ss(line);
        double x, y;
        ss >> x >> y;

        // Add the y value to the vector.
        y_values.push_back(y);
    }

    // Close the file.
    file.close();

    // Return the vector of y values.
    return y_values;
}

// Function to load the frequency info
std::vector<int> load_frequency_info(const std::string &filename)
{
    // Open the file.
    std::ifstream file(filename);

    // Check if the file opened successfully.
    if (!file) {
        // Throw an exception if the file could not be opened.
        throw std::runtime_error("Could not open file!");
    }

    // Create a vector to store the frequency info.
    std::vector<int> frequency_info;

    // Read each line from the file.
    for (std::string line; std::getline(file, line);) {
        // Split the line into two parts, the x value and the y value.
        std::stringstream ss(line);
        double range;
        int frequency;
        ss >> range >> frequency;

        // Add the frequency value to the frequency_info.
        frequency_info.push_back(frequency);
    }

    // Close the file.
    file.close();

    // Return the vector of frequency info.
    return frequency_info;
}

// compute the corresponding normalized value
double normalize_value(double value, double mean, double std_dev)
{
    return (value - mean) / std_dev;
}


// Function to get the Gaussian value
double get_Gaussian(double normalized_value)
{
    return (1 / sqrt(2 * M_PI)) * exp(-0.5 * pow(normalized_value, 2));
}

// Function to get the skewness
double get_skewness(double normalized_value, double alpha, std::vector<double> &lookup_table) {
    double abs_val = abs(alpha * normalized_value);
    double erf = 0.0f;
    double sign = normalized_value > 0 ? 1.0f : -1.0f;
    if (abs_val > 4.0) {
        erf = 1.0;
    } else {
        int look_up_index = static_cast<int>(std::round(abs_val * 100));
        erf = lookup_table[look_up_index];
    }
    return (1 + sign * erf);
}

// Merge function of coefficients
double merge_coefficients(double row_coefficient, double pixel_coefficient, double ratio_coefficient)
{
    return pow(pow(row_coefficient, 1.0) * pow(pixel_coefficient, 1.0) * pow(ratio_coefficient, 1.0), 1.0 / 3.0);
}
