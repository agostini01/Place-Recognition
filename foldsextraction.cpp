#include "foldsextraction.h"

FoldsExtraction::FoldsExtraction(const cv::Mat &imageDescriptors,
                                 const cv::Mat &imageLabels,
                                 const unsigned int &numberOfDifferentClasses)
    :   m_numberOfDifferentClasses(numberOfDifferentClasses)
    ,   m_foldsSize(imageDescriptors.rows/10)
    ,   m_folds(std::vector<Fold>())
{
    std::vector<int> indices;
    for(int i=0; i<imageLabels.rows; ++i)
    {
        indices.push_back(i);
    }

    std::random_device device;
    std::mt19937 generator(device());
    std::shuffle(indices.begin(), indices.end(), generator);

    cv::Mat imageDescriptorsShuffled(
        imageDescriptors.rows,
        imageDescriptors.cols,
        CV_32FC1
    );
    cv::Mat imageLabelsShuffled(
        imageLabels.rows,
        imageLabels.cols,
        CV_32FC1
    );

    for(int i=0; i<imageDescriptors.rows; ++i)
    {
        for(int j=0; j<imageDescriptors.cols; ++j)
        {
            imageDescriptorsShuffled.at<float>(i, j) = imageDescriptors.at<float>(indices[i], j);
        }
        imageLabelsShuffled.at<float>(i) = imageLabels.at<float>(indices[i]);
    }

    std::vector<std::pair<cv::Mat, unsigned int>> files;
    for (int i = 0; i < imageDescriptorsShuffled.rows; ++i)
    {
        files.push_back(std::make_pair(imageDescriptorsShuffled.row(i), imageLabelsShuffled.at<float>(i)));
    }


    while (!files.empty())
    {
        Fold aFold;
        for (unsigned int i = 0; i != m_foldsSize; ++i)
        {
            aFold.pushBackImage(files.back());
            files.pop_back();
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




