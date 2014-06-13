#include "crossvalidation.h"

CrossValidation::CrossValidation(FoldsExtraction &folds)
    :   m_folds(folds)
    ,   m_results(std::vector<cv::Mat>())
{
    /* Initialize the results with zeros Mat objects */
    for (int i = 0; i < m_folds.getNumberOfDifferentClasses(); ++i) {
        m_results.push_back(cv::Mat::zeros(m_folds.getNumberOfDifferentClasses(),
                                                     m_folds.getNumberOfDifferentClasses(),
                                                     CV_32F));
    }
}

float CrossValidation::crossValidadeSVM()
{
    std::cout<<m_folds.getFoldsSize()<<" is the folds std size"<<std::endl;
    std::cout<<" Peforming SVM Cross validation "<<std::endl;
    for( unsigned int foldNumber = 0; foldNumber != 1; ++foldNumber)
    {
        std::cout<<"Building SVM Classifier and Starting prediction to fold N: "<<foldNumber<<std::endl;
        SvmClassifier svm(m_folds);
        auto foldsIt = m_folds.begin();

        for(auto it = (foldsIt+foldNumber)->begin(); it != (foldsIt+foldNumber)->end(); ++it)
        {
//            std::cout<<&it->first<<std::endl;
            std::cout<<"Predicted "<< svm.predict(it->first);
            std::cout<<"\t Actual    "<< it->second<<std::endl<<std::endl;
        }
    }
}
