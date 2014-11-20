#ifndef __DETECTION_HAND_
#define __DETECTION_HAND_
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "utils.h"
using namespace std;

static const int lower_bound_skin[] = {0, 48, 80};
static const int upper_bound_skin[] = {0, 255, 255};

const int THRESHOLD_DEPTH_VALUE = 100;
const int max_binary_value = 255;
// helper function
//
//
static cv::Size Normalized_Filter(128, 128);
cv::Mat detect_hand_rgb(const cv::Mat& _input);
cv::Mat detect_hand_depth(const cv::Mat& _input);
cv::Mat normalize_image(const cv::Mat& _image);
#endif
