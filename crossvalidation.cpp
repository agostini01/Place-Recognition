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
    for (unsigned int i = 0; i < m_accuracyVector.size(); ++i)
    {
        meanAccuracy+=m_accuracyVector.at(i);
    }
    meanAccuracy= meanAccuracy/m_accuracyVector.size();

    return meanAccuracy;
}

float CrossValidation::getClassSensivity(const int &classNumber) const
{
//    float classAccuracy=0;
//    for (int j = 0; j < m_results.rows; ++j) {
//        classAccuracy+= (m_results.at<float>(j,classNumber));
//    }
//    classAccuracy=m_results.at<float>(classNumber,classNumber)/classAccuracy;
//    return classAccuracy;
        return getClassTP(classNumber)/(getClassP(classNumber));
}

float CrossValidation::getClassSpecificity(const int &classNumber) const
{
    return getClassTN(classNumber)/(getClassFP(classNumber)+getClassTN(classNumber));
}

float CrossValidation::getClassPrecision(const int &classNumber) const
{
    return getClassTP(classNumber)/(getClassTP(classNumber)+getClassFP(classNumber));
}

float CrossValidation::getClassNegativePredictedValue(const int &classNumber) const
{
    return getClassTN(classNumber)/(getClassTN(classNumber)+getClassFN(classNumber));
}

float CrossValidation::getClassFallOut(const int &classNumber) const
{
    return getClassFP(classNumber)/(getClassFP(classNumber)+getClassTN(classNumber));
}

float CrossValidation::getClassFalseDiscoveryRate(const int &classNumber) const
{
    return 1-getClassPrecision(classNumber);
}

float CrossValidation::getClassFalseNegativeRate(const int &classNumber) const
{
    return getClassFN(classNumber)/(getClassFN(classNumber)+getClassTP(classNumber));
}

float CrossValidation::getClassAccuracy(const int &classNumber) const
{
    return (getClassTP(classNumber)+getClassTN(classNumber))/(getClassP(classNumber)+getClassN(classNumber));
}

float CrossValidation::getClassF1Score(const int &classNumber) const
{
    return (2*getClassTP(classNumber))/(2*getClassTP(classNumber)+getClassFP(classNumber)+getClassFN(classNumber));
}

float CrossValidation::getClassP(const int &classNumber) const
{
    float positives = 0;

    for (int i = 0; i < m_results.rows; ++i)
    {
        positives+= (m_results.at<float>(i,classNumber));
    }
    return positives;
}

float CrossValidation::getClassN(const int &classNumber) const
{
    float negatives = 0;

    for (int i = 0; i < m_results.rows; ++i)
        for (int j = 0; j < m_results.cols; ++j) {
        {
            if(j != classNumber)
                negatives+= (m_results.at<float>(i,j));
        }
    }
    return negatives;
}

float CrossValidation::getClassTP(const int &classNumber) const
{
    return m_results.at<float>(classNumber,classNumber);
}

float CrossValidation::getClassTN(const int &classNumber) const
{
    float TN = 0;

    for (int j = 0; j < m_results.cols; ++j) {
        for (int i = 0; i < m_results.rows; ++i)
        {
            if(i != classNumber && j != classNumber)
                TN+= (m_results.at<float>(i,j));
        }
    }
    return TN;
}

float CrossValidation::getClassFP(const int &classNumber) const
{
    float FP = 0;
    for (int i = 0; i < m_results.rows; ++i)
    {
        if(i != classNumber)
            FP+= (m_results.at<float>(i,classNumber));
    }
    return FP;
}

float CrossValidation::getClassFN(const int &classNumber) const
{
    float FN = 0;
    for (int j = 0; j < m_results.cols; ++j)
    {
        if(j != classNumber)
            FN+= (m_results.at<float>(classNumber,j));
    }
    return FN;
}

void CrossValidation::displayResults()
{
    std::cout<<"Presenting results:"<<std::endl;
    std::cout<<"Class Accuracy:"<<std::endl;
    for (int i = 0; i < m_results.cols; ++i)
    {
        std::cout<<"___________________________________________________"<<std::endl;
        std::cout<<"Class "<<i<<std::endl;
        std::cout<<"Sensitivity (recall):\t\t\t"<<getClassSensivity(i)<<std::endl;
        std::cout<<"Specificity (true negative rate:\t"<<getClassSpecificity(i)<<std::endl;
        std::cout<<"Precision:\t\t\t\t"<<getClassPrecision(i)<<std::endl;
        std::cout<<"Negative Predicted Value:\t\t"<<getClassNegativePredictedValue(i)<<std::endl;
        std::cout<<"Fall-out (false positive rate:\t\t"<<getClassFallOut(i)<<std::endl;
        std::cout<<"False Discovery rate:\t\t\t"<<getClassFalseDiscoveryRate(i)<<std::endl;
        std::cout<<"False Negative rate (miss rate):\t"<<getClassFalseNegativeRate(i)<<std::endl;
        std::cout<<"Accuracy:\t\t\t\t"<<getClassAccuracy(i)<<std::endl;
        std::cout<<"F1 Score:\t\t\t\t"<<getClassF1Score(i)<<std::endl;
        std::cout<<std::endl;
    }

    std::cout<<std::endl;
    std::cout<<"Row \"i\" Display The classification for Class \"i\""<<std::endl;
    std::cout<<"eg.: RowxColumn"
               "\n0x0 class 0 classified as 0"
               "\n0x1 class 0 classified as 1"
               "\n0x2 class 0 classified as 2"<<std::endl;
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
    for (unsigned int i = 0; i < m_accuracyVector.size(); ++i)
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
