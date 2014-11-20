#ifndef __CLASSIFICATION_HAND_
#define __CLASSIFICATION_HAND_

#include "utils.h"
#include <iostream>
#include <string>
#include <vector>

using namespace std;

// use BAYES CLASSIFIER to classify hand pose
void train(cMatRef _training_data, cMatRef _label );
int predict(const CvSVM& bayes, const cv::Mat& _mat);
int predict(const CvNormalBayesClassifier& _bayes, const cv::Mat& _mat) ;
#endif
