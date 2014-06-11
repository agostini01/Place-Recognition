#ifndef SVMCLASSIFIER_H
#define SVMCLASSIFIER_H

#include "engine.h"     //  OPENCV Functions
#include <memory>       //  uniquePtr

class SvmClassifier
{
public:
    SvmClassifier();
    SvmClassifier(cv::Mat& trainingData, cv::Mat& trainingLabels);

    float predict(cv::Mat& descriptors) const;
private:
    CvSVMParams getCvSVMParams();

    /**
     * @brief m_svm
     * Only one SVM classifier will be created
     */
    std::unique_ptr<CvSVM> m_svm;
};

#endif // SVMCLASSIFIER_H
