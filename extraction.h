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
const double theta[] = {0, PI/2.0, PI, PI * 1.50};

const double sigma[] = {2.0, 3.0, 4.0, 5.0};

const int WIDTH_CELL= 8;
const int HEIGHT_CELL = 8;
const int WIDTH_GAUSSIAN= 8;
extern vector<cv::Mat> exp_matrix;
void init_exp_matrix();
cv::Mat extract(ConMatRef _input);
#endif
