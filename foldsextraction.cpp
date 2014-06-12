#include "foldsextraction.h"

FoldsExtraction::FoldsExtraction(const cv::Mat &imageDescriptors,
                                 const cv::Mat &imageDescriptorsClasses,
                                 const unsigned int &numberOfDifferentClasses)
    :   m_numberOfDifferentClasses(numberOfDifferentClasses)
    ,   m_foldsSize(imageDescriptors.rows/10)
    ,   m_folds(std::vector<Fold>())
{
    // Creat a single fold - store everything in  vector
    std::vector<std::pair<cv::Mat, unsigned int>> files;
    for (int i = 0; i < imageDescriptors.rows; ++i)
    {
        files.push_back(std::make_pair(imageDescriptors.row(i), imageDescriptorsClasses.at<float>(i)));
    }


    // Separete data in 10 folds randomly (folds may have different number files per class
    srand (time(NULL));
    while (!files.empty())
    {
        Fold aFold;
        for (unsigned int i = 0; i < m_foldsSize; ++i)
        {
            auto it = files.begin();
            int randomAccess = rand()%files.size();
            aFold.pushBackImage(*(it+randomAccess));
            files.erase(it+randomAccess);
        }

        m_folds.push_back(aFold);
    }
    std::cout<<m_folds.size()<<std::endl;
}
std::vector<Fold> FoldsExtraction::getFolds() const
{
    return m_folds;
}

void FoldsExtraction::setFolds(const std::vector<Fold> &folds)
{
    m_folds = folds;
}

