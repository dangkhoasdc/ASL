#include "extraction.h"

vector<cv::Mat> exp_matrix;
void init_exp_matrix() {
    float squared_width_gaussian = 2 * WIDTH_GAUSSIAN * WIDTH_GAUSSIAN;
    for (int i = 0; i < HEIGHT_CELL; i++) {
        for (int j = 0; j < WIDTH_CELL; j++) {
            float xi = 16*i - 8;
            float yj = 16*j - 8;
            cv::Mat value(cv::Size(128,128), CV_32FC1);
            for (int ii = 0; ii < 128; ii++) {
                for (int jj = 0; jj < 128; jj++) {
                    value.at<float>(ii,jj) = exp(-(pow(ii - xi, 2.0) + pow(jj - yj,2.0))/squared_width_gaussian);
                }
            }
            exp_matrix.push_back(value);
        }
    }
}
cv::Mat extract(ConMatRef _input) {
    vector<cv::Mat> responses;
    cv::Mat c;
    // filter an input image to 16 Gabor filters
    //cout << "size of theta : " << SIZEOF_ARR(theta) << endl;
    //cout << "size of sigma: " << SIZEOF_ARR(sigma) << endl;

    for (unsigned int i = 0; i < SIZEOF_ARR(theta); ++i) {
        for (unsigned int j = 0 ; j < SIZEOF_ARR(sigma); ++j) {
            cv::Mat result;
            cv::Mat kernel = cv::getGaborKernel(cv::Size(5,5),
                    sigma[j],
                    theta[i],
                    10.0,
                    0.5
                    );
            cv::filter2D(_input,
                    result,
                    _input.depth(),
                    kernel);
            cv::imshow("debug gabor filter", result);
            cv::waitKey(0);
            responses.push_back(result);
        }
    }
    cv::Size response_size = responses[0].size();

    //float squared_width_gaussian = 2 * WIDTH_GAUSSIAN * WIDTH_GAUSSIAN;
    for (int k = 0; k < 16; k++) {
        cv::Mat cell(HEIGHT_CELL, WIDTH_CELL,CV_32FC1);
        for (int i = 0; i < HEIGHT_CELL; i++) {
        for (int j = 0; j < WIDTH_CELL; j++) {
            float cell_value = 0;
            //float xi = 16 * i - 8;
            //float yj = 16 * j - 8;
            for (int ii = 0; ii < response_size.height; ii++) {
            for (int jj = 0; jj < response_size.width; jj++) {
                cell_value += responses[k].at<float>(ii,jj)
                    //*exp(-(pow(ii - xi, 2.0) + pow(jj - yj,2.0))/squared_width_gaussian);
                    * exp_matrix[i*WIDTH_CELL+ j].at<float>(ii,jj);
            }
            }
            cell.at<float>(i,j) = cell_value;
        }
        }
        c.push_back(cell);
    }
    // release all filtered images
    responses.clear();
    return c.reshape(0,1);
}
