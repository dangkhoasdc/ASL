#include "utils.h"
#include "detection.h"
#include "extraction.h"
#include "classification.h"
using namespace cv;
void showHistogram(Mat& img) {
	int bins = 256;             // number of bins
	int nc = img.channels();    // number of channels

	vector<Mat> hist(nc);       // histogram arrays

	// Initalize histogram arrays
	for (int i = 0; i < hist.size(); i++)
		hist[i] = Mat::zeros(1, bins, CV_32SC1);

	// Calculate the histogram of the image
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			for (int k = 0; k < nc; k++)
			{
				uchar val = nc == 1 ? img.at<uchar>(i,j) : img.at<Vec3b>(i,j)[k];
				hist[k].at<int>(val) += 1;
			}
		}
	}

	// For each histogram arrays, obtain the maximum (peak) value
	// Needed to normalize the display later
	int hmax[3] = {0,0,0};
	for (int i = 0; i < nc; i++)
	{
		for (int j = 0; j < bins-1; j++)
			hmax[i] = hist[i].at<int>(j) > hmax[i] ? hist[i].at<int>(j) : hmax[i];
	}

	const char* wname[3] = { "blue", "green", "red" };
	Scalar colors[3] = { Scalar(255,0,0), Scalar(0,255,0), Scalar(0,0,255) };

	vector<Mat> canvas(nc);

	// Display each histogram in a canvas
	for (int i = 0; i < nc; i++)
	{
		canvas[i] = Mat::ones(125, bins, CV_8UC3);

		for (int j = 0, rows = canvas[i].rows; j < bins-1; j++)
		{
			line(
				canvas[i],
				Point(j, rows),
				Point(j, rows - (hist[i].at<int>(j) * rows/hmax[i])),
				nc == 1 ? Scalar(200,200,200) : colors[i],
				1, 8, 0
			);
		}

		imshow(nc == 1 ? "value" : wname[i], canvas[i]);
	}
}
void training_phase(const string& _filename);
void testing_phase(const string& _filename);
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
void training_phase(const string& _filename) {
    vector<InfoData> training;
    readfile(_filename, training);
    cv::Mat training_set;
    cv::Mat label_set;
    bool test = false;
    int id = 0;
    for (unsigned int i = 0; i < training.size(); i++) {
        cv::Mat feature_vector;

        string rgb_filename = training[i].first;
        cout << "["<<id++<<"]:"<<rgb_filename << flush;

        string depth_filename = training[i].first.replace(training[i].first.find("color"), 5, "depth");

        cv::Mat color_image_2 = cv::imread(rgb_filename.c_str(),1);
        cv::Mat color_image;
        cv::cvtColor(color_image_2, color_image, CV_BGR2GRAY);
        cv::equalizeHist(color_image, color_image);
        cv::Mat detected_color_image = normalize_image(color_image);
        color_image.release();
        color_image_2.release();
        feature_vector.push_back(extract(detected_color_image));

        detected_color_image.release();

        cv::Mat depth_image_2 = cv::imread(depth_filename.c_str(),1);
        cv::Mat depth_image;
        cv::cvtColor(depth_image_2, depth_image, CV_BGR2GRAY);
        cv::equalizeHist(depth_image, depth_image);
        cv::Mat detected_depth_image = normalize_image(depth_image);
        depth_image.release();
        feature_vector.push_back(extract(detected_depth_image));
        detected_depth_image.release();

        cv::Mat normalized_vector = feature_vector.reshape(1,1);
        if (test == false) {
            cout << "size: " << normalized_vector.size() << endl;
            cout << "mat:" << feature_vector<< endl;
            test = true;
        }
        feature_vector.release();
        training_set.push_back(normalized_vector);
        label_set.push_back(training[i].second - 'a' + 1);
        cout << "....completed" << endl;
        normalized_vector.release();
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
    int id = 0;
    CvSVM svm;
    svm.load("model.yml");
    //CvNormalBayesClassifier bayes;
    //bayes.load("model.yml");
    for (unsigned int i = 0; i < testing.size(); i++) {
        cv::Mat feature_vector;
        string rgb_filename = testing[i].first;
        cout << "["<<id++<<"]:"<<rgb_filename << flush << endl;
        string depth_filename = testing[i].first.replace(testing[i].first.find("color"), 5, "depth");

        cv::Mat color_image_2 = cv::imread(rgb_filename.c_str(),1);
        cv::Mat color_image;
        cv::cvtColor(color_image_2, color_image, CV_BGR2GRAY);
        cv::equalizeHist(color_image, color_image);
        cv::Mat detected_color_image = normalize_image(color_image);
        feature_vector.push_back(extract(detected_color_image));
        detected_color_image.release();
        color_image.release();
        color_image_2.release();

        cv::Mat depth_image_2 = cv::imread(depth_filename.c_str(),1);
        cv::Mat depth_image;
        cv::cvtColor(depth_image_2, depth_image, CV_BGR2GRAY);
        cv::equalizeHist(depth_image, depth_image);
        cv::Mat detected_depth_image = normalize_image(depth_image);
        feature_vector.push_back(extract(detected_depth_image));



        cv::Mat normalized_vector = feature_vector.reshape(1,1);
        int result = predict(svm, normalized_vector);
        //int result = predict(bayes, normalized_vector);
        if (result == (testing[i].second - 'a' + 1)) accuracy++;
        normalized_vector.release();
    }

    cout << "result = " << (float) accuracy / testing.size() << endl;

}
