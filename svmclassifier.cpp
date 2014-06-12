#include "svmclassifier.h"

/**
 * @brief SvmClassifier::SvmClassifier
 * Default constructor initializes SVM Classifier with the file under:
 * ../output/svm_filename
 */
SvmClassifier::SvmClassifier()
    :   m_svm(new CvSVM())
{
    m_svm->load("../output/svm_filename"); // loading
}

/**
 * @brief SvmClassifier::SvmClassifier
 * It creates the classifier based on the acquired training data
 * Saves the classfier to ../output/svm_filename
 * @param trainingData
 * is a cv::Mat(NumberOfFilesToPredict,NumberOfFeatures,CV_32FC1)
 * @param trainingLabels
 * is a cv::Mat(NumberOfFilesToPredict,1,CV_32FC1)
 */
SvmClassifier::SvmClassifier(cv::Mat &trainingData, cv::Mat &trainingLabels)
    :   m_svm(new CvSVM())
{
    //  1) Initializing the training and label matrix
    cv::Mat trainingSVM_mat = trainingData;
    cv::Mat labelsSVM_mat = trainingLabels;

    //  2) Set SVM parameters (for kernel separation - instead of 2d separation line)
    CvSVMParams params = getCvSVMParams();

    //  3) Training SVM classifier
    m_svm->train(trainingSVM_mat, labelsSVM_mat, cv::Mat(), cv::Mat(), params);

    //  4) Saving Classfier for further usage
    m_svm->save("../output/svm_filename");
}

SvmClassifier::SvmClassifier(const FoldsExtraction &myFolds, const unsigned int &foldToNotInclude)
{
    cv::Mat trainingSVM_mat;
    cv::Mat labelsSVM_mat;

    unsigned int count =0;
    for(auto it = myFolds.getFolds().begin(); it != myFolds.getFolds().end(); ++it)
    {
        if((it) != (it+foldToNotInclude)) // if it is not the test fold
        {
            for(auto it2 = it->getFold().begin(); it2 != it->getFold().end(); ++it2)
            {
                trainingSVM_mat.row(count) = it2->first;
                labelsSVM_mat.row(count) = it2->second;
            }
        }
    }

    //  2) Set SVM parameters (for kernel separation - instead of 2d separation line)
    CvSVMParams params = getCvSVMParams();

    //  3) Training SVM classifier
    m_svm->train(trainingSVM_mat, labelsSVM_mat, cv::Mat(), cv::Mat(), params);

    //  4) Saving Classfier for further usage
    std::string outputPath = "../output/svm_filename";
    outputPath.append(std::to_string(foldToNotInclude));
    m_svm->save(outputPath.c_str());
}

/**
 * @brief SvmClassifier::predict
 * Interface for OPENCV'cvSVM.predict() method
 * @param descriptors
 * is a cv::Mat(1,NumberOfFeatures,CV_32FC1)
 * @return
 * the predicted class considering the labels
 */
float SvmClassifier::predict(cv::Mat &descriptors) const
{
    return m_svm->predict(descriptors);
}

/**
 * @brief SvmClassifier::getCvSVMParams
 * may be changed to tune values and SVM methods
 * @return
 * CvSVMParams class with tunned values
 */
CvSVMParams SvmClassifier::getCvSVMParams() const
{
    CvSVMParams params = CvSVMParams();
    params.svm_type = CvSVM::C_SVC;
    params.kernel_type = CvSVM::RBF; //CvSVM::RBF, CvSVM::LINEAR ...
    params.degree = 0; // for poly
    params.gamma = 3; // for poly/rbf/sigmoid
    params.coef0 = 0; // for poly/sigmoid

    params.C = 7; // for CV_SVM_C_SVC, CV_SVM_EPS_SVR and CV_SVM_NU_SVR
    params.nu = 0.0; // for CV_SVM_NU_SVC, CV_SVM_ONE_CLASS, and CV_SVM_NU_SVR
    params.p = 0.0; // for CV_SVM_EPS_SVR

    params.class_weights = NULL; // for CV_SVM_C_SVC
    params.term_crit.type = CV_TERMCRIT_ITER +CV_TERMCRIT_EPS;
    params.term_crit.max_iter = 1000;
    params.term_crit.epsilon = 1e-6;

    return params;
}
