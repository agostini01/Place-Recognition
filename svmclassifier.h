#ifndef SVMCLASSIFIER_H
#define SVMCLASSIFIER_H

#include "engine.h"     //  OPENCV Functions
#include "fold.h"
#include "foldsextraction.h"

class SvmClassifier
{
public:
    SvmClassifier();
    SvmClassifier(cv::Mat& trainingData, cv::Mat& trainingLabels);
    SvmClassifier(FoldsExtraction &myFolds, const unsigned int &foldToNotInclude);
    SvmClassifier(FoldsExtraction &myFolds);

    float predict(cv::Mat& descriptor) const;
private:
    CvSVMParams getCvSVMParams() const;

    /**
     * @brief m_svm
     * A smart pointer to SVM classifier will be created
     * Smart pointers has built-in garbage collector
     * cv::Ptr<> bahaves the same as C++11 shared pointer
     */
    cv::Ptr<CvSVM> m_svm;


};

#endif // SVMCLASSIFIER_H
