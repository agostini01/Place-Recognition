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
    std::cout<<"Training Done"<<std::endl;
    //  4) Saving Classfier for further usage
    m_svm->save("../output/svm_classifier.xml");
}

/**
 * @brief SvmClassifier::SvmClassifier
 * Special Constructor to be used with cross Validation
 * @param myFolds
 * The 10 Folds in a single object
 * @param foldToNotInclude
 * The number of the fold we want to classify
 */
SvmClassifier::SvmClassifier(FoldsExtraction &myFolds, const unsigned int &foldToNotInclude)
    :   m_svm(new CvSVM())
{
    //  1) Adapting folds to SVM input (every descriptor in a single matrix)
    cv::Mat trainingSVM_mat((myFolds.getNumberOfDescriptors()-myFolds.getNumberOfDescriptors()/10), myFolds.getNumberOfFeatures(), CV_32FC1);
    cv::Mat labelsSVM_mat = cv::Mat::zeros((myFolds.getNumberOfDescriptors()-myFolds.getNumberOfDescriptors()/10),1,CV_32FC1);

    unsigned int count =0; // To count how many descriptors are being added
    for(auto it = myFolds.begin(); it != myFolds.end(); ++it)   // fold access
    {
        if((it) != (myFolds.begin()+foldToNotInclude)) // if it is not the test fold
        {
            for(auto it2 = it->begin(); it2 != it->end(); ++it2)    // pair access
            {
                for (int i = 0; i < it2->first.cols; ++i)
                {
                    trainingSVM_mat.at<float>(count,i) = it2->first.at<float>(i);
                }
                labelsSVM_mat.at<float>(count) = it2->second;
                ++count;
            }
        }
    }

    //  2) Set SVM parameters (for kernel separation - instead of 2d separation line)
    CvSVMParams params = getCvSVMParams();

    //  3) Training SVM classifier
    m_svm->train(trainingSVM_mat, labelsSVM_mat, cv::Mat(), cv::Mat(), params);

    //  4) Saving Classfier for further usage
    std::string outputPath = "../output/svm_crosvalidation_";
    outputPath.append(std::to_string(foldToNotInclude));
    m_svm->save(outputPath.c_str());
}

SvmClassifier::SvmClassifier(FoldsExtraction &myFolds)
    :   m_svm(new CvSVM())
{
    //  1) Adapting folds to SVM input (every descriptor in a single matrix)
    cv::Mat trainingSVM_mat(myFolds.getNumberOfDescriptors(), myFolds.getNumberOfFeatures(), CV_32F);
    cv::Mat labelsSVM_mat(1,myFolds.getNumberOfDescriptors(),CV_32FC1);

    unsigned int count =0; // To count how many descriptors are being added
    for(auto it = myFolds.begin(); it != myFolds.end(); ++it)   // fold access
    {
        if(true) // if it is not the test fold
        {
            for(auto it2 = it->begin(); it2 != it->end(); ++it2)    // pair access
            {
                for (int i = 0; i < it2->first.cols; ++i)
                {
                    trainingSVM_mat.at<float>(count,i) = it2->first.at<float>(i);
                }
                labelsSVM_mat.at<float>(count) = it2->second;
                ++count;
            }
        }
    }

    //  2) Set SVM parameters (for kernel separation - instead of 2d separation line)
    CvSVMParams params = getCvSVMParams();

    //  3) Training SVM classifier
    m_svm->train(trainingSVM_mat, labelsSVM_mat, cv::Mat(), cv::Mat(), params);

    //  4) Saving Classfier for further usage
    std::string outputPath = "../output/svm_crosvalidation_";
    outputPath.append(std::to_string(100));
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
float SvmClassifier::predict(cv::Mat &descriptor) const
{
    return m_svm->predict(descriptor);
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
