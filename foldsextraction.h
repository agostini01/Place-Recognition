#ifndef FOLDSEXTRACTION_H
#define FOLDSEXTRACTION_H

#include "engine.h"
#include "fold.h"       // to hold a fold instance
#include <utility>      // std::pair -> Folds hold pair of descriptors and class
#include <vector>       // to hold several folds
#include <iostream>

#include <stdlib.h>     // srand, rand -> To randomize the folds
#include <time.h>       // time

/**
 * @brief The FoldsExtraction class
 * Class resposable to perform data spliting to perform cross validation
 * It is meant to receive descriptors of images.
 * In order to make it work properly those descriptors (stored in m_descriptors)
 * has to be in groups of equal size.
 * Each group has descriptors of a single class.
 * The descriptors classes are stored in a row of m_descriptorClasses as integers
 */
class FoldsExtraction
{
public:
    FoldsExtraction(const cv::Mat &imageDescriptors,
                    const cv::Mat &imageLabels,
                    const unsigned int &getNumberOfDifferentClasses);

    std::vector<Fold> getFolds() const;
    void setFolds(const std::vector<Fold> &getFolds);

    unsigned int getNumberOfDifferentClasses() const;

    unsigned int getFoldsSize() const;

    unsigned int getNumberOfFeatures() const;

    unsigned int getNumberOfDescriptors() const;

    typedef std::vector<Fold>::iterator iterator;
    iterator begin() { return m_folds.begin(); }
    iterator end() { return m_folds.end(); }

private:
    /**
     * @brief m_numberOfDifferentClasses
     * How many different classes on the descriptors
     */
    const unsigned int m_numberOfDifferentClasses;

    /**
     * @brief m_foldsSize
     * Holds the avarege folds size (number of imges per fold)
     */
    const unsigned int m_foldsSize;

    /**
     * @brief m_Folds
     * Will hold 10 different folds where crossvalidation may be performed
     */
    std::vector<Fold> m_folds;
};

#endif // FOLDSEXTRACTION_H
