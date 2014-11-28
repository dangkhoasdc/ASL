#include "utils.h"
#include "detection.h"
#include "extraction.h"
#include "classification.h"
using namespace cv;
void training_phase(const string& _filename);
void testing_phase(const string& _filename);
// just process extraction process
cv::Mat process_img(const string& _filename);
// read filename of data and process all stages
cv::Mat process_data(const string& _filename);
int main(int argc, char const* argv[])
{
    init_exp_matrix();
    switch (argc) {
        case 2:
            testing_phase(argv[1]);
            break;
        case 3:
            training_phase(argv[1]);
            testing_phase(argv[2]);
            break;
        default:
            cerr << "Failure argument of program " << endl;
            break;
    }
    return 0;
}
cv::Mat process_img(const string& _filename) {
    cv::Mat image = cv::imread(_filename.c_str(),1);
    cv::Mat gray_img;
    cv::cvtColor(image, gray_img, CV_BGR2GRAY);
    // pre-process image by histogram equalization
    cv::equalizeHist(gray_img, gray_img);
    cv::Mat detected_color_image = normalize_image(gray_img);

    image.release();
    gray_img.release();
    return extract(detected_color_image);
}

cv::Mat process_data(const string& _filename) {
    string filename(_filename);
    cv::Mat feature_vector;
    // read and process color image
    string rgb_filename = filename;

    feature_vector.push_back(process_img(rgb_filename));

    // read ad process depth image
    string depth_filename = filename.replace(filename.find("color"), 5, "depth");
    feature_vector.push_back(process_img(depth_filename));
    cv::Mat normalize_image = feature_vector.reshape(1,1);
    cv::norm(normalize_image, normalize_image);
    return normalize_image;
}

void training_phase(const string& _filename) {
    vector<InfoData> training;
    readfile(_filename, training);
    cv::Mat training_set;
    cv::Mat label_set;
    for (unsigned int i = 0; i < training.size(); i++) {
        cout << "["<<i<<"]:" << training[i].first << flush;
        // add feature vector to training set
        training_set.push_back(process_data(training[i].first));
        label_set.push_back(training[i].second - 'a');
        cout << "....completed" << endl;
    }
    //cout << "label:" << label_set << endl;
    cout << "Training ..." << flush;
    train_svm(training_set,label_set);
    cout << "Completed" << endl;
}
void testing_phase(const string& _filename) {
    vector<InfoData> testing;
    readfile(_filename, testing);
    cv::Mat testing_set;
    cv::Mat label_set;
    int accuracy = 0;
    CvSVM svm;
    svm.load("model.yml");
    //CvNormalBayesClassifier bayes;
    //bayes.load("model.yml");
    for (unsigned int i = 0; i < testing.size(); i++) {
        cout << "["<<i<<"]:" << testing[i].first << flush;
        cv::Mat feature_vector = process_data(testing[i].first);
        int result = predict(svm, feature_vector);
        //int result = predict(bayes, normalized_vector);
        if (result == (testing[i].second - 'a')) accuracy++;
        feature_vector.release();
        cout<< endl;
    }
    cout << "result = " << (float) accuracy / testing.size() << endl;

}
