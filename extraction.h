#ifndef __EXTRACTION_HAND_
#define __EXTRACTION_HAND_
#include "utils.h"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
const double gabor_lambda = 16;
const double PI = 3.1415;
// TODO: dummy values -> choose reasonable values
const double theta[] = {0, PI/2.0, PI, 4.0/3 * PI};
const double sigma[] = {2, 3, 4, 5};

const int WIDTH_CELL= 8;
const int HEIGHT_CELL = 8;
const int WIDTH_GABOR_FILTER = 8;

cv::Mat extract(ConMatRef _input);
#endif
