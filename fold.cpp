#include "fold.h"

Fold::Fold()
    :   m_theFold(std::vector<std::pair<cv::Mat, unsigned int> >())
{
}

std::vector<std::pair<cv::Mat, unsigned int> > Fold::getFold() const
{
    return m_theFold;
}

void Fold::setTheFold(const std::vector<std::pair<cv::Mat, unsigned int> > &theFold)
{
    m_theFold = theFold;
}

void Fold::pushBackImage(const cv::Mat &imageDescriptors, const unsigned int &imageClass)
{
    m_theFold.push_back(std::make_pair(imageDescriptors, imageClass));
}

void Fold::pushBackImage(const std::pair<cv::Mat, unsigned int> &thePair)
{
    m_theFold.push_back(thePair);
}


unsigned int Fold::size() const
{
    return m_theFold.size();
}

