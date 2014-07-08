#ifndef CROSSVALIDATION_H
#define CROSSVALIDATION_H

#include "foldsextraction.h"
#include "svmclassifier.h"
#include "engine.h"
#include <vector>
#include <iomanip>          //To display results

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

    //For results display
    void displayResults();
    float getMeanAccuracy();

    /**
     *  For the next functions the following terminology
     * found on wikipedia will be used:
     *
     *  true positive (TP)
     * eqv. with hit
     *
     *  true negative (TN)
     * eqv. with correct rejection
     *
     *  false positive (FP)
     * eqv. with false alarm, Type I error
     *
     *  false negative (FN)
     * eqv. with miss, Type II error
     */

    /**
     * @brief getClassSensivity
     * @param classNumber
     * @return TPR = TP/(TP+FN)
     */
    float getClassSensivity(const int &classNumber) const;

    /**
     * @brief getClassSpecificity
     * @param classNumber
     * @return SPC = TN/(FP+TN)
     */
    float getClassSpecificity(const int &classNumber) const;

    /**
     * @brief getClassPrecision
     * @param classNumber
     * @return PPV = TP/(TP +FP)
     */
    float getClassPrecision(const int &classNumber) const;

    /**
     * @brief getClassNegativePredictedValue
     * @param classNumber
     * @return NPV = TN/(TN + FN)
     */
    float getClassNegativePredictedValue(const int &classNumber) const;

    /**
     * @brief getClassFallOut
     * @param classNumber
     * @return FPR = FP/(FP+TN)
     */
    float getClassFallOut(const int &classNumber) const;

    /**
     * @brief getClassDiscoveryRate
     * @param classNumber
     * @return FDR = 1-PPV
     */
    float getClassFalseDiscoveryRate(const int &classNumber) const;

    /**
     * @brief getClassFalseDiscoveryRate
     * @param classNumber
     * @return FNR = FN/(FN + TP)
     */
    float getClassFalseNegativeRate(const int &classNumber) const;

    /**
     * @brief getClassAccuracy
     * @param classNumber
     * @return ACC = (TP + TN)/(P + N)
     */
    float getClassAccuracy(const int &classNumber) const;

    /**
     * @brief getClassF1Score
     * @param classNumber
     * @return 2TP/(2TP + FP + FN)
     */
    float getClassF1Score(const int &classNumber) const;




private:
    /**
     * @brief m_folds
     * Will hold the 10 folds
     */
    FoldsExtraction m_folds;

    /**
     * @brief m_results
     * A NxN matriz that holds results for each classification performed
     * The corrected classfied are on the main  diagonal
     */
    cv::Mat m_results;

    /**
     * @brief m_accuracyVector
     * Holds the accuracy for each fold
     * Vector legth = number of folds
     */
    std::vector<float> m_accuracyVector;


    // To get class information from confusion matrix
    float getClassTP(const int &classNumber) const;
    float getClassTN(const int &classNumber) const;
    float getClassFP(const int &classNumber) const;
    float getClassFN(const int &classNumber) const;
    float getClassP(const int &classNumber) const;
    float getClassN(const int &classNumber) const;

};

#endif // CROSSVALIDATION_H
