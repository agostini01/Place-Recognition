#ifndef CROSSVALIDATION_H
#define CROSSVALIDATION_H

#include "foldsextraction.h"
#include "svmclassifier.h"
#include "engine.h"
#include <vector>

/**
 * @brief The CrossValidation class
 * Is meant to perform a 10 fold cross validation
 * It depends on the classifiers constructors
 * This class object is meant to be created with a 10 fold object
 * that will be used to cross validade any type of classfier (SVM, Bayes, Randon trees)
 */
class CrossValidation
{
public:
    CrossValidation(FoldsExtraction &folds);

    float crossValidadeSVM();



private:
    /**
     * @brief m_folds
     * Will hold the 10 folds
     */
    FoldsExtraction m_folds;

    /**
     * @brief m_results
     * Is a vector of NxN matriz that holds results for each classification performed
     * The corrected classfied are on the main  diagonal
     */
    std::vector<cv::Mat> m_results;


};

#endif // CROSSVALIDATION_H
