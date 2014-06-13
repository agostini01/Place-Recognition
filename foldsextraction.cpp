#include "foldsextraction.h"

FoldsExtraction::FoldsExtraction(const cv::Mat &imageDescriptors,
                                 const cv::Mat &imageLabels,
                                 const unsigned int &numberOfDifferentClasses)
    :   m_numberOfDifferentClasses(numberOfDifferentClasses)
    ,   m_foldsSize(imageDescriptors.rows/10)
    ,   m_folds(std::vector<Fold>())
{
    // Creat a single fold - store everything in  vector
    std::vector<std::pair<cv::Mat, unsigned int>> files;
    for (int i = 0; i < imageDescriptors.rows; ++i)
    {
        files.push_back(std::make_pair(imageDescriptors.row(i), imageLabels.at<float>(i)));
    }


    // Separete data in 10 folds randomly (folds may have different number files per class
    srand (time(NULL));
    while (!files.empty())
    {


        Fold aFold;
        for (unsigned int i = 0; i != m_foldsSize; ++i)
        {
            auto it = files.begin();
            int randomAccess = rand()%files.size();
            aFold.pushBackImage(*(it+randomAccess));
            files.erase(it+randomAccess);
        }

        m_folds.push_back(aFold);
    }
}
std::vector<Fold> FoldsExtraction::getFolds() const
{
    return m_folds;
}

void FoldsExtraction::setFolds(const std::vector<Fold> &folds)
{
    m_folds = folds;
}
unsigned int FoldsExtraction::getNumberOfDifferentClasses() const
{
    return m_numberOfDifferentClasses;
}
unsigned int FoldsExtraction::getFoldsSize() const
{
    return m_foldsSize;
}

unsigned int FoldsExtraction::getNumberOfFeatures() const
{
    return m_folds.at(0).getNumberOfFeatures();
}

unsigned int FoldsExtraction::getNumberOfDescriptors() const
{
    return m_foldsSize*10;
}




