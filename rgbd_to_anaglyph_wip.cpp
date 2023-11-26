// import numpy as np
// from PIL import Image
// import cv2
// import matplotlib.pyplot as plt
// import argparse
// from tqdm import tqdm
// from collections import deque, namedtuple
// import time
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;

struct Params {
    double f;
    double cx;
    double cy;
    double distance_between_eyes;
};

vector<double> deproj(double x, double y, double Z, Params params) {
    double X = (x-params.cx)*Z/params.f;
    double Y = (y-params.cy)*Z/params.f;
    return {X, Y, (double)Z};
}


pair<cv::Mat, cv::Mat> construct_right_image(cv::Mat img_left, cv::Mat depth_image, Params params) {
    int height = depth_image.rows;
    int width = depth_image.cols;
    cv::Mat mask = cv::Mat::ones(height, width, CV_8UC1) * 255;

    cv::Mat img_right = cv::Mat::zeros(img_left.size(), CV_8UC3);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x ++) {
            double Z = depth_image.at<uint16_t>(y, x) / 1000.0; // Convert to meters

            if (Z == 0) {
                img_right.at<int>(y, x) = img_left.at<int>(y, x);
                mask.at<int>(y, x) = 0;
                continue;
            }
            cout << "Z: " << Z <<" " << depth_image.at<short>(y, x) << endl;
            vector<double> deprojected = deproj(x, y, Z, params);
            double X = deprojected[0];
            double Y = deprojected[1];
            // Z = deprojected[2];
            double X_ = X - params.distance_between_eyes;
            double x_ = params.f * X_ + params.cx * Z;
            double y_ = params.f * Y + params.cy * Z;
            x_ /= Z;
            y_ /= Z;
            if (0 <= round(x_) && round(x_) < width && 0 <= round(y_) && round(y_) < height) {
                int x__ = round(x_);
                int y__ = round(y_);
                // cout << "x__: " << x__ << " y__: " << y__ << endl;
                // cout << "x: " << x << " y: " << y << endl;
                img_right.at<int>(y__, x__) = img_left.at<int>(y, x);
                mask.at<int>(y__, x__) = 0;
                cout << x_ << " " <<x<< endl;
            }
        }
    }
    // Print mask shape
    cout << "Mask shape: " << mask.size() << endl;
    cv::imshow("Mask", mask);

    cv::imshow("Left image", img_left);
    cv::imshow("Right image", img_right);
    cv::waitKey(0);
    return {img_right, mask};
}

int main(int argc, char** argv) {
    cv::Mat img_left = cv::imread("car.jpg");
    cv::Mat img_depth = cv::imread("car.png", cv::IMREAD_UNCHANGED | cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
    Params params = {525.0, 319.5, 239.5, 0.05};
    pair<cv::Mat, cv::Mat> res = construct_right_image(img_left, img_depth, params);
    cv::Mat img_right = res.first;
    cv::Mat mask = res.second;
    cv::imshow("Left image", img_left);
    cv::imshow("Right image", img_right);
    cv::Mat img_left_anaglyph; 
    std::vector<cv::Mat> channels;
    cv::split(img_left, channels);
    channels[0].setTo(cv::Scalar(0)); // Set the blue channel to 0
    channels[1].setTo(cv::Scalar(0)); // Set the green channel to 0
    cv::merge(channels, img_left_anaglyph);

    cv::Mat img_right_anaglyph; 
    std::vector<cv::Mat> channels2;
    cv::split(img_right, channels2);
    channels2[2].setTo(cv::Scalar(0)); // Set the red channel to 0
    cv::merge(channels2, img_right_anaglyph);
    cv::imshow("Left anaglyph", img_left_anaglyph);
    cv::imshow("Right anaglyph", img_right_anaglyph);

    cv::Mat img_anaglyph = img_left_anaglyph + img_right_anaglyph;
    cv::imshow("Anaglyph", img_anaglyph);

    // Print min and max values of img_right
    cv::Point min_loc, max_loc;
    double minn, maxx;
    cv::minMaxLoc(img_depth, &minn, &maxx, &min_loc, &max_loc);
    std::cout << "Min: " << minn <<" Max: " << maxx << std::endl;
    // cv::imshow("Mask", mask);
    cv::waitKey(0);
    std::cout << "Hello World!" << std::endl;
    return 0;
}
