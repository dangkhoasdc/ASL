#ifndef __UTILS_ASL_SYSTEM
#define __UTILS_ASL_SYSTEM

#define SIZEOF_ARR(a) sizeof(a)/sizeof(a[0])
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <utility>
#undef DEBUG_MODE
enum READFILE_RESULT {
    SUCCESS,
    CANNOT_READ_FILE,
    ERROR_FORMAT
};
using namespace std;
typedef pair<string,char>   InfoData;
typedef const cv::Mat&      cMatRef;
typedef vector<InfoData>    DataSet;
typedef const cv::Mat&      ConMatRef;
typedef const cv::Mat&      cMatRef;

// helper functions
READFILE_RESULT readfile(const string& _filename, DataSet& _infodata);
void showHistogram(Mat& img) ;
#endif
