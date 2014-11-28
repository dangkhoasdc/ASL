#include "classification.h"

void train_svm(cMatRef _training_data, cMatRef _label ) {
    CvSVMParams *m_param;
    CvSVM *m_Svm;

    m_param = new CvSVMParams();
    m_Svm = new CvSVM();
    m_param->svm_type = CvSVM::NU_SVC;
    m_param->kernel_type = CvSVM::RBF;

    //m_param->degree = 0;
    //m_param->gamma = 0.0001;
    //m_param->coef0 = 0;

    //m_param->C = 1;
    m_param->nu = 0.0099;
    //m_param->p = 0.0;

    m_param->class_weights = NULL;
    m_param->term_crit.type = CV_TERMCRIT_ITER + CV_TERMCRIT_EPS;
    m_param->term_crit.max_iter = 100;
    m_param->term_crit.epsilon = 0.1;

    m_Svm->train_auto(_training_data, _label, cv::Mat(), cv::Mat(), *m_param);
    m_Svm->save("model.yml");
    if (m_param)
        delete m_param;
    if (m_Svm)
        delete m_Svm;
}
void train(cMatRef _training_data, cMatRef _label ) {
	//CvSVMParams *m_param;
	//CvSVM *m_Svm;

	//m_param = new CvSVMParams();
	//m_Svm = new CvSVM();
	//m_param->svm_type = CvSVM::NU_SVC;
	//m_param->kernel_type = CvSVM::RBF;
	//m_param->degree = 0;
	//m_param->gamma = 0.0001;
	//m_param->coef0 = 0;

	//m_param->C = 1;
	//m_param->nu = 0.0099;
	//m_param->p = 0.0;

	//m_param->class_weights = NULL;
	//m_param->term_crit.type = CV_TERMCRIT_ITER + CV_TERMCRIT_EPS;
	//m_param->term_crit.max_iter = 1000;
	//m_param->term_crit.epsilon = FLT_EPSILON;

	//m_Svm->train_auto(_training_data, _label, cv::Mat(), cv::Mat(), *m_param, 24);

	//if (m_param)
		//delete m_param;
	//if (m_Svm)
		//delete m_Svm;
    CvNormalBayesClassifier bayes;
    bayes.train(_training_data, _label);
    bayes.save("model.yml");
}
int predict(const CvSVM& _svm, const cv::Mat& _mat) {
    return (int) _svm.predict(_mat);
}
int predict(const CvNormalBayesClassifier& _bayes, const cv::Mat& _mat) {
    return (int) _bayes.predict(_mat);
}


