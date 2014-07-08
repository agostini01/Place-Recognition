/*

 * main.cpp
 *
 *  Created on: 06/03/2014
 *      Author: agostini
 *
 http://stackoverflow.com/questions/11602577/bag-of-words-training-and-testing-opencv-matlab
 To describe a proper bag of visual words the following steps will
 be follow

 1 - Extract the SIFT local feature vectors from your set of images;
 2 - Put all this local feature vectors into a single set. At this
 point you don't even need to store from which image each local
 feature vector was extracted;
 3 - Apply a clustering algorithm (e.g. k-means) over the set of
 local feature vectors in order to find centroid coordinates and
 assign an id to each centroid. This set of centroids will be your
 vocabulary;
 4 - The global feature vector will be a histogram that counts how
 many times each centroid occurred in each image. To compute the
 histogram find the nearest centroid for each local feature vector.
 */

#include "engine.h"
#include "examples.h"
#include "DataParser.h"
#include "datahandler.h"
#include "foldsextraction.h"
#include "crossvalidation.h"
#include "svmclassifier.h"
#include <map>
#include <list>
#include <iterator>


#include <iostream>

using namespace std;
using namespace cv;

/**
 * @brief makePathToImage
 * @param pathToCategory
 * @param imageName
 * @return the image path in a string. Windows '\' path divider can be changed here
 */
string makePathToImage(const std::string& pathToCategory, const std::string& imageName)
{
    std::string pathToImage;
    pathToImage.clear();
    pathToImage.append(pathToCategory);
    pathToImage.append("/");    // For linux path syntax
    pathToImage.append(imageName);

    return pathToImage;
}

int main(int argc, char **argv)
{

    std::cout<<"Starting Application"<<std::endl;
//    examples example;
//    example.gnuplotHistExample();
//    example.gnuplotExample();
//    example.historiogramCalculationExample(argv[2]);
//    example.bowExtractorExample(argv[1], argv[2]);

    dataHandler handler1;
    std::map<std::string, std::list<std::string> > trainClassAndFiles=handler1.buildDataFiles("../data/train/");
    std::map<std::string, std::list<std::string> > testClassAndFiles=handler1.buildDataFiles("../data/test/");
//        cv::Mat img1 = imread(argv[1]);
//        cv::Mat img2 = imread(argv[2]);

    cv::Mat imgToAdd;
    cv::Mat* imgPointer;
    dataParser::DataParser parser;
    //Creates Detectors Extractors Matchers
    cv::Ptr<cv::FeatureDetector> featureDetector =
            cv::FeatureDetector::create("SURF");
    cv::Ptr<cv::DescriptorExtractor> descExtractor =
            cv::DescriptorExtractor::create("SURF");
    cv::Ptr<cv::DescriptorMatcher> descMatcher =
            cv::DescriptorMatcher::create("BruteForce");
    std::vector<cv::KeyPoint> imgKeypoints;

    //Creates the dictionary trainer and its vocabulary
    cv::TermCriteria tc(1000, 10, 0.1);
    int dictionarySize = 20; // It is wise to keep this number close to the number of possible classes
    int retries = 1;
    int flags = cv::KMEANS_PP_CENTERS;
    cv::BOWKMeansTrainer bowTrain(dictionarySize, tc, retries, flags);
    cv::Ptr<cv::BOWImgDescriptorExtractor> bowExtractor;
    bowExtractor = new cv::BOWImgDescriptorExtractor(descExtractor,
            descMatcher);
    cv::Mat vocabulary;

    /**
    * If True it computes the Vocabulary used to get smaller descriptors (takes time)
    * If False it loads the previous computed vocabulary
    */
    if(false)
    {
        //Start to handle images
        for (auto it = trainClassAndFiles.begin(); it != trainClassAndFiles.end(); ++it)
        {
            std::cout<<std::endl<<"Creating BOW for category: "<<it->first<<std::endl;
            std::cout<<"Processig images..."<<std::endl;
            std::string pathToCategory;
            pathToCategory.clear();
            pathToCategory.append("../data/train/");
            pathToCategory.append(it->first);
            for (auto it2 = it->second.begin(); it2 != it->second.end(); ++it2)
            {
                imgToAdd = imread(makePathToImage(pathToCategory, *it2));
                imgPointer = &imgToAdd;
                //Adds image descriptors to the trainer
                cv::Mat imgDescriptors;
                featureDetector->detect(*imgPointer, imgKeypoints); //Find keypoints
                descExtractor->compute(*imgPointer, imgKeypoints, imgDescriptors); // Create Descriptors
                bowTrain.add(imgDescriptors); //Add descriptors to trainer
            }
            std::cout<<std::endl;
        }
        //Set vocabulary with trainer
        vocabulary = bowTrain.cluster();
        parser.writeVocabulary(vocabulary,"../output/cvVocabulary.xml");
    }
    else
    {
        std::cout<<"Reading vocabulary from file"<<std::endl;
        vocabulary=parser.readVocabulary("../output/cvVocabulary.xml");
    }
    bowExtractor->setVocabulary(vocabulary);

    //  EXECUTING feature extraction
    cv::Mat descriptors;
    cv::Mat newDescriptors;
    vector<vector<int> > pointIdxsOfClusters;
    cv::Mat trainingDescriptors(1500,dictionarySize,CV_32FC1);
    cv::Mat trainingLabels = cv::Mat::zeros(1500,1,CV_32FC1);

    /**
    * If True it executes image feature extraction, Bow, and files re-write (it takes time)
    * If False it loads the already claculated descriptors and labels
    */
    if(false)
    {
        int i = 0;
        for (auto it = trainClassAndFiles.begin(); it != trainClassAndFiles.end(); ++it)
        {
            std::cout<<std::endl<<"Adding category: "<<it->first<<" train matrix"<<std::endl;
            std::cout<<"Processig images..."<<std::endl;
            std::string pathToCategory;
            pathToCategory.clear();
            pathToCategory.append("../data/train/");
            pathToCategory.append(it->first);
            for (auto it2 = it->second.begin(); it2 != it->second.end(); ++it2)
            {
                imgToAdd = imread(makePathToImage(pathToCategory, *it2));
                featureDetector->detect(imgToAdd,imgKeypoints);
                bowExtractor->compute(imgToAdd, imgKeypoints, newDescriptors,
                        &pointIdxsOfClusters, &descriptors);

                std::cout<<"-> "<<trainingLabels.at<float>(i,0)<<"\t";
                for (int j = 0; j < trainingDescriptors.cols; ++j)
                {
                    trainingDescriptors.at<float>(i,j)=newDescriptors.at<float>(j);
                    std::cout<<trainingDescriptors.at<float>(i,j)<<"\t";
                }
                std::cout<<std::endl;
                ++i;
            }
        }

        if( remove( "../output/trainingSet.xml" ) != 0 )
        {
            std::cout<<"Error deleting file: ../output/trainingSet.xml"<<std::endl;
        }
        else
        {
            std::cout<<"../output/trainingSet.xml successfully deleted"<<std::endl;
        }
        parser.writeMat(trainingDescriptors,"../output/trainingSet.xml");

        if( remove( "../output/trainingSetLabels.xml" ) != 0 )
        {
            std::cout<<"Error deleting file: ../output/trainingSetLabels.xml"<<std::endl;
        }
        else
        {
            std::cout<<"../output/trainingSetLabels.xml successfully deleted"<<std::endl;
        }
        parser.writeMat(trainingLabels,"../output/trainingSetLabels.xml");
    }
    else
    {
        std::cout<<"Reading training descriptors from file"<<std::endl;
        trainingDescriptors = parser.readMat("../output/trainingSet.xml");
        std::cout << "Training Descriptors Size"
                << " is: " << trainingDescriptors.size() << std::endl;

        std::cout<<"Reading training descriptors Labels from file"<<std::endl;
        trainingLabels = parser.readMat("../output/trainingSetLabels.xml");
        std::cout << "Training Labels Size"
                << " is: " << trainingLabels.size() << std::endl<< std::endl;
    }

    // Cross Validation Rotine
    unsigned int numberOfClasses = 15;
    FoldsExtraction my10Folds = FoldsExtraction(trainingDescriptors, trainingLabels, numberOfClasses);


    // Cross Validation SVM
    CrossValidation myCrossValidation(my10Folds);
    std::cout<<"Startin SVM crossvalidaion"<<std::endl;
    myCrossValidation.crossValidadeSVM();
    myCrossValidation.displayResults();



    /**
    * If True it executes video prediction through webcan
    * If False do nothing
    */
    if(true)
    {
        // Check and open video feed
        cv::VideoCapture videoFeed;
        if (videoFeed.isOpened())
        {
            videoFeed.release();
        }
        videoFeed.open(0);

        //set height and width of capture frame
        videoFeed.set(CV_CAP_PROP_FRAME_WIDTH, 640);
        videoFeed.set(CV_CAP_PROP_FRAME_HEIGHT, 480);


        std::cout<<"Performing new prediction"<<std::endl;
        std::cout<<"Training new SVM Classifier"<<std::endl;
        SvmClassifier mySvmClassifier(trainingDescriptors,trainingLabels);

        std::cout<<"Starting video prediction..."<<std::endl;
        std::cout<<"Press ESC on the new window to exit"<<std::endl;
        cv::namedWindow("Prediction", CV_WINDOW_AUTOSIZE);
        cv::Mat textImage = 255*cv::Mat::ones(50,400,CV_32F);
        while(waitKey(60) != 1048603)
        {
            videoFeed.read(imgToAdd);
            featureDetector->detect(imgToAdd,imgKeypoints);
            bowExtractor->compute(imgToAdd, imgKeypoints, newDescriptors,
                    &pointIdxsOfClusters, &descriptors);

            // To write predicted class
            textImage = 255*cv::Mat::ones(50,400,CV_32F);
            cv::putText(textImage,"Predicted Class   " + std::to_string(mySvmClassifier.predict(newDescriptors) )
                          ,cv::Point(10,30),100,1,0);

            cv::imshow("Prediction",textImage);
        }

        if (videoFeed.isOpened())
        {
            videoFeed.release();
        }
    }

    delete imgPointer;
    return 0;
}


