#include "detection.h"


// use thresholding method to determine a hand region
cv::Mat detect_hand_rgb(const cv::Mat& _input) {
// convert a input image to HSV space

    cv::Mat converted_img;
    cv::Mat detected_skin;
    cv::Mat output;

    vector<int> lower(lower_bound_skin, lower_bound_skin + SIZEOF_ARR(lower_bound_skin));
    vector<int> upper(upper_bound_skin, upper_bound_skin + SIZEOF_ARR(upper_bound_skin));

    cv::cvtColor(_input, converted_img, CV_BGR2HSV);
    cv::inRange(converted_img, lower, upper, detected_skin);
    cv::resize(converted_img, output, Normalized_Filter);

    return output;
}


// use thresholding method based on depth value to determine a hand region
cv::Mat detect_hand_depth(const cv::Mat& _input) {
// convert to a binary image
    cv::Mat binary;
    cv::Mat output;
    binary = _input;
    cv::threshold(binary, output, THRESHOLD_DEPTH_VALUE, max_binary_value, 0);

    cv::resize(output, output, Normalized_Filter);
    return output;
}

cv::Mat normalize_image(const cv::Mat& _image) {
    cv::Mat output;
    cv::resize(_image, output, Normalized_Filter);
    //output.convertTo(output, CV_32FC1, 1.0/255.0);
    output.convertTo(output, CV_32FC1 );
    return output;
}
