//
// Created by Etch on 6/14/2023.
//
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <string>

struct Point {
    double x;
    double y;
};

double fit_func(double x, double alpha, double k) {
    return std::exp(-alpha * x) * std::pow((x / 0.1 + 1), -k);
}

int main() {
    double alpha = 0.139958;
    double k = 0.09;
    double max_x = 0;
    double max_y = 0;
    unsigned int cnt_total = 0;
    unsigned int cnt_valid = 0;
    std::string line;
    std::vector<Point> points;

    std::ifstream file("D:\\Repo\\UATD\\Dataset\\Training\\Attenuation_data_720.txt");

    if (file.is_open()) {
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            double x, y;
            ss >> x >> y;
            cnt_total++;

            // find the max x value
            if (x > max_x) {
                max_x = x;
            }
            // find the max y value
            if (y > max_y) {
                max_y = y;
            }

            // condition for outliers
            if (y <= 9 * fit_func(x, alpha, k)) {
                points.push_back({x, y});
                cnt_valid++;
            }
        }

        file.close();
    } else {
        std::cout << "Unable to open file";
        return 1;
    }

    std::ofstream outfile("D:\\Repo\\UATD\\Dataset\\Training\\outlier_removed_720.txt");

    if (outfile.is_open()) {
        for (const auto &point : points) {
            outfile << point.x << " " << point.y << "\n";
        }

        outfile.close();
    } else {
        std::cout << "Unable to open output file";
        return 1;
    }

    // print the max x value, total points and valid points number
    std::cout << "Max distance value: " << max_x << std::endl;
    std::cout << "Max relative intensity value: " << max_y << std::endl;
    std::cout << "Valid points: " << cnt_valid << std::endl;
    std::cout << "Total points: " << cnt_total << std::endl;

    return 0;
}
