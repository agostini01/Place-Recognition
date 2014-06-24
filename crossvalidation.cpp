#include "crossvalidation.h"

CrossValidation::CrossValidation(FoldsExtraction &folds)
    :   m_folds(folds)
    ,   m_results(cv::Mat::zeros(m_folds.getNumberOfDifferentClasses(),
                                 m_folds.getNumberOfDifferentClasses(),
                                 CV_32F))
    ,   m_accuracyVector(std::vector<float>())
{
//    /* Initialize the results with zeros Mat objects */
//    for (unsigned int i = 0; i < m_folds.getNumberOfDifferentClasses(); ++i) {
//        m_results.push_back(cv::Mat::zeros(m_folds.getNumberOfDifferentClasses(),
//                                                     m_folds.getNumberOfDifferentClasses(),
//                                                     CV_32F));
//    }
}

float CrossValidation::getMeanAccuracy()
{
    float meanAccuracy=0;
    for (int i = 0; i < m_accuracyVector.size(); ++i)
    {
        meanAccuracy+=m_accuracyVector.at(i);
    }
    meanAccuracy= meanAccuracy/m_accuracyVector.size();

    return meanAccuracy;
}

float CrossValidation::getClassAccuracy(const int &classNumber) const
{
    float classAccuracy=0;
    for (int j = 0; j < m_results.rows; ++j) {
        classAccuracy+= (m_results.at<float>(classNumber,j));
    }
    classAccuracy=m_results.at<float>(classNumber,classNumber)/classAccuracy;
    return classAccuracy;
}

void CrossValidation::displayResults()
{
    std::cout<<"Presenting results:"<<std::endl;
    std::cout<<"Class Accuracy:"<<std::endl;
    for (int i = 0; i < m_results.cols; ++i)
    {
        std::cout<<"Class "<<i<<": "<<getClassAccuracy(i)<<std::endl;
    }

    std::cout<<std::endl;
    std::cout<<"Collum \"i\" Display The classification for Class \"i\""<<std::endl;
    for (int i = 0; i != m_results.rows; ++i)
    {
        for (int j = 0; j != m_results.cols; ++j)
        {
            std::cout << '|' << std::setw(3) << m_results.at<float>(i,j);
        }
        std::cout<<'|'<<std::endl;
    }
    std::cout<<std::endl;

    std::cout<<"Fold Accuracy Result"<<std::endl;
    for (int i = 0; i < m_accuracyVector.size(); ++i)
    {
        std::cout<<"Fold "<<i<<": "<< m_accuracyVector.at(i)<<std::endl;
    }


    std::cout<<std::endl;
    std::cout<<"Mean Accuracy:"<<std::endl;
    std::cout<<getMeanAccuracy()<<std::endl;
}

float CrossValidation::crossValidadeSVM()
{
    std::cout<<"Peforming SVM Cross validation "<<std::endl;
    auto foldsBeginIt = m_folds.begin();
    float predicted = 0;
    std::cout<<"Building SVM Classifier and Starting prediction"<<std::endl;
    std::cout<<"Finishing classifications for fold: ";
    for( unsigned int foldNumber = 0; foldNumber != 10; ++foldNumber)
    {
        int correctedClassified = 0;
        int totalClassified = 0;

        SvmClassifier svm(m_folds,foldNumber);
        for(auto it = (foldsBeginIt+foldNumber)->begin(); it != (foldsBeginIt+foldNumber)->end(); ++it)
        {
            ++totalClassified;
            predicted = svm.predict(it->first);
            if (fabs((predicted - it->second)) < 0.01)
            {
                ++m_results.at<float>(predicted,predicted);
                ++correctedClassified;
            }
            else
            {
                ++m_results.at<float>(it->second,predicted);
            }

        }

        m_accuracyVector.push_back(static_cast<float>(correctedClassified)/totalClassified);
        std::cout<<foldNumber<<" ";
    }
    std::cout<<std::endl;

    return 0;
}
