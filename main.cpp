#include "utils.h"
#include "detection.h"
#include "extraction.h"
#include "classification.h"


void training_phase(const string& _filename);
void testing_phase(const string& _filename);
int main(int argc, char const* argv[])
{
    init_exp_matrix();
    training_phase(argv[1]);
    testing_phase(argv[2]);
    return 0;
}
void training_phase(const string& _filename) {
    vector<InfoData> training;
    readfile(_filename, training);
    cv::Mat training_set;
    cv::Mat label_set;
    bool test = true;
    int id = 0;
    for (unsigned int i = 0; i < training.size(); i++) {
        string rgb_filename = training[i].first;
        cout << "["<<id++<<"]:"<<rgb_filename << flush;
        string depth_filename = training[i].first.replace(training[i].first.find("color"), 5, "depth");

        cv::Mat color_image_2 = cv::imread(rgb_filename.c_str(),1);
        cv::Mat color_image;
        cv::cvtColor(color_image_2, color_image, CV_BGR2GRAY);
        cv::equalizeHist(color_image, color_image);
        cv::Mat depth_image_2 = cv::imread(depth_filename.c_str(),1);
        cv::Mat depth_image;
        cv::cvtColor(depth_image_2, depth_image, CV_BGR2GRAY);
        cv::equalizeHist(depth_image, depth_image);
        //cv::Mat detected_color_image = detect_hand_rgb(color_image);
        //cv::Mat detected_depth_image = detect_hand_depth(depth_image);

        cv::Mat detected_color_image = normalize_image(color_image);
        cv::Mat detected_depth_image = normalize_image(depth_image);
        color_image.release();
        depth_image.release();
        cv::Mat feature_vector;

        feature_vector.push_back(extract(detected_color_image));
        feature_vector.push_back(extract(detected_depth_image));

        detected_color_image.release();
        detected_depth_image.release();
        cv::Mat normalized_vector = feature_vector.reshape(1,1);
        if (test == false) {
            cout << "size: " << normalized_vector.size() << endl;
            cout << "mat:" << feature_vector<< endl;
            test = true;
        }
        feature_vector.release();
        training_set.push_back(normalized_vector);
        label_set.push_back((float)training[i].second - 'a');
        cout << "....completed" << endl;
        normalized_vector.release();
    }
    //cout << "label:" << label_set << endl;
    cout << "Training ..." << flush;
    train(training_set,label_set);
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
    cout << "completed 24" << endl;
    for (unsigned int i = 0; i < testing.size(); i++) {
        string rgb_filename = testing[i].first;
        string depth_filename = testing[i].first.replace(testing[i].first.find("color"), 5, "depth");

        cv::Mat color_image_2 = cv::imread(rgb_filename.c_str(),1);
        cv::Mat color_image;
        cv::cvtColor(color_image_2, color_image, CV_BGR2GRAY);
        cv::equalizeHist(color_image, color_image);

        cv::Mat depth_image_2 = cv::imread(depth_filename.c_str(),1);
        cv::Mat depth_image;
        cv::cvtColor(depth_image_2, depth_image, CV_BGR2GRAY);
        cv::equalizeHist(depth_image, depth_image);

        cv::Mat detected_color_image = normalize_image(color_image);
        cv::Mat detected_depth_image = normalize_image(depth_image);

        cv::Mat feature_vector;

        feature_vector.push_back(extract(detected_color_image));
        feature_vector.push_back(extract(detected_depth_image));

        cv::Mat normalized_vector = feature_vector.reshape(1,1);

        int result = predict(svm, normalized_vector);
        //int result = predict(bayes, normalized_vector);
        if (result == (testing[i].second - 'a')) accuracy++;
    }

    cout << "result = " << (float) accuracy / testing.size() << endl;

}
