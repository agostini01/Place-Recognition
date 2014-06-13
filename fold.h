#ifndef FOLD_H
#define FOLD_H

#include <utility>  // Pair -> to store descriptor and class
#include <vector>   // To hold vector of pairs (the actual fold)
#include <engine.h> // OpenCV stuff

class Fold
{
public:
    Fold();
    std::vector<std::pair<cv::Mat, unsigned int> > getFold() const;
    void setTheFold(const std::vector<std::pair<cv::Mat, unsigned int> > &theFold);
    void pushBackImage(const cv::Mat& imageDescriptors, const unsigned int& imageClass);
    void pushBackImage(const std::pair<cv::Mat,unsigned int> &thePair);
    unsigned int size() const;

    unsigned int getNumberOfFeatures() const;

    typedef std::vector<std::pair<cv::Mat, unsigned int>>::iterator iterator;
    iterator begin() { return m_theFold.begin(); }
    iterator end() { return m_theFold.end(); }


private:
    /**
     * @brief m_theFold
     * holds pairs of image's Descriptors and image's class
     */
    std::vector<std::pair<cv::Mat, unsigned int>> m_theFold;
};

#endif // FOLD_H
