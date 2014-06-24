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

//    virtual ~SvmClassifier()
//    {
////        std::cout<<"Called destructor"<<std::endl;
//        if(m_svm != nullptr)
//        {
////            std::cout<<"Deleting m_svm"<<std::endl;
//            m_svm.delete_obj();
//        }
////        std::cout<<"Assigning m_svm to a nullptr"<<std::endl;
//        m_svm = nullptr;
//    }

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
